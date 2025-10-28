import torch
import torch.distributed as dist
from torch.optim import AdamW
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig
from my_fsdp_utils import make_model_fsdp, save_fsdp_lora, save_fsdp_optimizer
import torch.distributed.fsdp
import torch.distributed.fsdp.api
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import argparse
from dreamomni2.pipeline_dreamomni2 import DreamOmni2Pipeline

def parse_args():
    """Parses command-line arguments for model paths and server configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vlm_path", 
        type=str, 
        default="/home/yanzhang/models/DreamOmni2/vlm-model", 
        help="Path to the VLM model directory."
    )
    parser.add_argument(
        "--edit_lora_path", 
        type=str, 
        default="/home/yanzhang/models/DreamOmni2/edit_lora", 
        help="Path to the FLUX.1-Kontext editing LoRA weights directory."
    )
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        default="/home/yanzhang/models/FLUX.1-Kontext-dev", 
        help="Path to the FLUX.1-Kontext editing."
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=32
    )
    
    args = parser.parse_args()
    return args

# 1️⃣ 初始化分布式环境
accelerator = Accelerator()
device = accelerator.device

# 2️⃣ 加载预训练模型
ARGS = parse_args()
vlm_path = ARGS.vlm_path
base_model = ARGS.base_model_path
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    vlm_path, torch_dtype="bfloat16", device_map="auto"
)
processor = AutoProcessor.from_pretrained(vlm_path)
pipe = DreamOmni2Pipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="balanced")

# 3️⃣ 注入 LoRA 层
lora_rank = ARGS.lora_rank
lora_alpha = ARGS.lora_alpha
lora_config = LoraConfig(
    r=lora_rank, lora_alpha=lora_alpha, target_modules=["q_proj", "v_proj"], lora_dropout=0.05
)
pipe.add_adapter(lora_config)

model = get_peft_model(pipe, lora_config)

# 4️⃣ 包裹为 FSDP 模型
fsdp_model = make_model_fsdp(
    model,
    param_dtype=torch.float16,
    device=device,
    sharding_strategy=torch.distributed.fsdp.api.ShardingStrategy.HYBRID_SHARD,
    part_size=1e7,
    use_orig_params=False
)

# 5️⃣ 设置优化器
optimizer = AdamW(fsdp_model.parameters(), lr=1e-4)

# 6️⃣ 加载数据 & 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        with accelerator.accumulate(fsdp_model):
            outputs = fsdp_model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
    accelerator.print(f"Epoch {epoch}: loss={loss.item()}")

# 7️⃣ 保存 LoRA 权重（仅主进程）
save_dir = "./lora_ckpt"
save_fsdp_lora(fsdp_model, save_dir, is_main_process=accelerator.is_main_process)

# 8️⃣ （可选）保存优化器状态
if accelerator.is_main_process:
    save_fsdp_optimizer({"model": fsdp_model}, optimizer, save_dir)
