import os
import re
import json
import gc
import functools
import contextlib
from typing import Dict, Union, Optional, Type, Set

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType,
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
import torch.distributed.checkpoint as torch_dcp
import torch.distributed.checkpoint.state_dict
from torch.distributed.fsdp.api import (
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
)
import accelerate
import safetensors
import diffusers
import transformers
from huggingface_hub.serialization import split_torch_state_dict_into_shards
import os, re, json
from typing import Union
import torch
import safetensors.torch
import accelerate
# from .ema_utils import EMAModel

# 精度管理与模块
def upcast_trainable_param_to_fp32_(fsdp_model):
    for m in FSDP.fsdp_modules(fsdp_model):
        if m._has_params:
            param = m._flat_param
            if (
                param.dtype != torch.float32
                and param.device != torch.device("meta")
                and param.requires_grad
            ):
                param.data = param.data.to(torch.float32)
                m._handle._orig_param_dtype = torch.float32


def get_module_to_ignore_mixed_precision():
    try:
        from apex.normalization import FusedLayerNorm

        return [
            torch.nn.GroupNorm,
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            FusedLayerNorm,
        ]
    except:
        return [
            torch.nn.GroupNorm,
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
        ]


def is_fsdp_model(model):
    return len(FSDP.fsdp_modules(model)) > 0

# FSDP自动包装策略
def size_based_auto_wrap_policy(
    module: torch.nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    # Additional custom arguments
    min_num_params: int = int(1e8),
    force_leaf_modules: Optional[Set[Type[torch.nn.Module]]] = None,
    exclude_wrap_modules: Optional[Set[Type[torch.nn.Module]]] = None,
) -> bool:
    """
    A size-based auto wrap policy.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.

        min_num_params (int): Customizable policy input that controls the size
            threshold over which a module is ready to be wrapped. This is in
            units of numel.
        force_leaf_modules (Set[Type[nn.Module]]): Set of module types to keep
            as leaves, i.e. their children will never be wrapped.
        exclude_wrap_modules (Set[Type[nn.Module]]): Set of module types to be
            excluded in wrapping.

    Returns:
        Whether ``module`` should be wrapped.
    """
    force_leaf_modules = (
        size_based_auto_wrap_policy.FORCE_LEAF_MODULES  # type: ignore[attr-defined]
        if force_leaf_modules is None
        else force_leaf_modules
    )
    exclude_wrap_modules = (
        size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES  # type: ignore[attr-defined]
        if exclude_wrap_modules is None
        else exclude_wrap_modules
    )

    # Keep the argument `min_num_params` for BC for now, but it represents the
    # minimum non-wrapped *numel* before triggering a wrapping
    min_nonwrapped_numel = min_num_params
    is_large = nonwrapped_numel >= min_nonwrapped_numel
    STOP_FLAG_NAME = "__FSDP_STOP_WARP_FLAG_CUSTOM_POLICY_size_based_auto_wrap_policy"
    if recurse:
        # use MixedPrecision cause ALWAYS recurse
        if isinstance(module, tuple(force_leaf_modules)):
            for m in module.children():
                m.apply(lambda m: setattr(m, STOP_FLAG_NAME, True))
        return True
    else:
        if getattr(module, size_based_auto_wrap_policy.LEAF_ROOT_FLAG_NAME, False):
            return True
        elif getattr(module, STOP_FLAG_NAME, False):
            return False
        else:
            # If we are not recursing, determine if we should wrap.
            return is_large and not isinstance(module, tuple(exclude_wrap_modules))


# Set those defaults to the size_based_auto_wrap_policy function. Make them easy to be imported.
size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES = {torch.nn.ModuleList, torch.nn.ModuleDict}  # type: ignore[attr-defined]
size_based_auto_wrap_policy.FORCE_LEAF_MODULES = {torch.nn.MultiheadAttention}  # type: ignore[attr-defined]
size_based_auto_wrap_policy.LEAF_ROOT_FLAG_NAME = (
    "__FSDP_LEAF_ROOT_FLAG_CUSTOM_POLICY_size_based_auto_wrap_policy"
)


def mark_leaf_root_(module):
    setattr(
        module,
        size_based_auto_wrap_policy.LEAF_ROOT_FLAG_NAME,
        True,
    )


def make_model_fsdp(
    model,
    param_dtype,
    device,
    reduce_dtype=None,
    buffer_dtype=None,
    sync_module_states=True,
    process_group=None,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    module_classes_to_ignore_mixed_precision=None,
    ignored_states=None,
    ignored_modules=None,
    auto_wrap_policy=None,
    part_size=1e6,
    force_leaf_modules=None,
    exclude_wrap_modules=None,
    use_orig_params=False
):
    if module_classes_to_ignore_mixed_precision is None:
        module_classes_to_ignore_mixed_precision = (
            get_module_to_ignore_mixed_precision()
        )
    if auto_wrap_policy is not None:
        auto_wrap_policy = auto_wrap_policy
    elif sharding_strategy == ShardingStrategy.NO_SHARD:
        auto_wrap_policy = None
    else:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=part_size,
            force_leaf_modules=force_leaf_modules,
            exclude_wrap_modules=exclude_wrap_modules,
        )

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        process_group=process_group,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        use_orig_params=use_orig_params,
        sync_module_states=sync_module_states,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype or torch.float32,
            buffer_dtype=buffer_dtype or torch.float32,
            keep_low_precision_grads=False,
            cast_forward_inputs=False,
            cast_root_forward_inputs=True,
            _module_classes_to_ignore=module_classes_to_ignore_mixed_precision,
        ),
        auto_wrap_policy=auto_wrap_policy,
        ignored_states=ignored_states,
        ignored_modules=ignored_modules,
        device_id=device,
    )
    torch.cuda.empty_cache()
    gc.collect()
    return model

# 模型加载与保存部分
def save_fsdp_lora(
    model_to_save,                      # FSDP 包裹的模型
    save_directory: Union[str, os.PathLike],
    is_main_process: bool = True,
    lora_regex: str = r"(?:lora)",  # 根据自己命名习惯调
):
    """
    仅保存 LoRA 层的权重。适用于 FSDP 并与 safetensors 兼容。
    """
    # 1. 解包 FSDP，拿到裸模型
    unwrapped_model = accelerate.utils.extract_model_from_parallel(model_to_save)

    # 2. 创建保存目录
    if is_main_process:
        os.makedirs(save_directory, exist_ok=True)

    # 3. 收集完整 state_dict（CPU 上）
    state_dict = torch_dcp.state_dict.get_model_state_dict(
        model_to_save,
        options=torch_dcp.state_dict.StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            ignore_frozen_params=False,
        ),
    )

    # 4. 过滤出 LoRA 参数
    lora_pattern = re.compile(lora_regex)
    lora_state_dict = {
        k: v for k, v in state_dict.items() if lora_pattern.search(k) is not None
    }

    if not lora_state_dict:
        raise ValueError(
            "未找到匹配 LoRA 的参数。请检查 lora_regex 是否符合命名规则。"
        )

    # 5. 保存为单文件 *.safetensors
    if is_main_process:
        weight_file = os.path.join(save_directory, "adapter_model.safetensors")
        safetensors.torch.save_file(
            lora_state_dict, weight_file, metadata={"format": "pt", "type": "lora"}
        )


def load_fsdp_model_(model_to_load: FSDP, save_directory: Union[str, os.PathLike]):
    with FSDP.state_dict_type(
        model_to_load,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(
            rank0_only=False,
        ),
    ):
        _model = model_to_load.from_pretrained(save_directory)
        model_to_load.load_state_dict(_model.state_dict())

# 优化器保存与加载部分
def save_fsdp_optimizer(
    models: Dict,
    optimizer_to_save: torch.optim.Optimizer,
    save_directory: Union[str, os.PathLike],
    is_main_process: bool = True,
):
    _fsdp_state_dict_config = dict(
        state_dict_type=StateDictType.FULL_STATE_DICT,
        optim_state_dict_config=FullOptimStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        ),
    )
    mgrs = list()
    for m in models.values():
        if len(FSDP.fsdp_modules(m)) > 0:
            mgrs.append(FSDP.state_dict_type(m, **_fsdp_state_dict_config))

    with contextlib.ExitStack() as stack:
        for mgr in mgrs:
            stack.enter_context(mgr)
        optim_state_dict = FSDP.optim_state_dict(
            torch.nn.ModuleDict(models),
            optimizer_to_save,
        )
        if is_main_process:
            torch.save(
                optim_state_dict, os.path.join(save_directory, "optim_states.pth")
            )


def load_fsdp_optimizer_(
    models: Dict,
    optimizer_to_load: torch.optim.Optimizer,
    save_directory: Union[str, os.PathLike],
):
    _fsdp_state_dict_config = dict(
        state_dict_type=StateDictType.FULL_STATE_DICT,
        optim_state_dict_config=FullOptimStateDictConfig(
            rank0_only=False,
        ),
    )
    mgrs = list()
    for m in models.values():
        if len(FSDP.fsdp_modules(m)) > 0:
            mgrs.append(FSDP.state_dict_type(m, **_fsdp_state_dict_config))

    with contextlib.ExitStack() as stack:
        for mgr in mgrs:
            stack.enter_context(mgr)
        optimizer_path = os.path.join(save_directory, "optim_states.pth")
        assert os.path.isfile(optimizer_path)
        optim_state_dict = torch.load(optimizer_path)
        optim_state_dict = FSDP.optim_state_dict_to_load(
            torch.nn.ModuleDict(models),
            optimizer_to_load,
            optim_state_dict,
        )
        optimizer_to_load.load_state_dict(optim_state_dict)