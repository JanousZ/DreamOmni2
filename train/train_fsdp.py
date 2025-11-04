import torch
import torch.distributed as dist
from torch.optim import AdamW
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import argparse
from dreamomni2.pipeline_dreamomni2 import DreamOmni2Pipeline
from utils.vprocess import process_vision_info
import numpy as np
from my_datasets.replace5k import Replace5kDataset
from torch.utils.data import DataLoader
import os
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, BackwardPrefetch
from accelerate.utils import ProjectConfiguration
import logging
from accelerate.logging import get_logger
import diffusers
import datasets
import transformers
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5Tokenizer,
    is_wandb_available
)
from tqdm import tqdm
from utils.infer_utils import _encode_prompt_with_t5, _encode_prompt_with_clip, encode_prompt
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")

# 防止tokenizer并行死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./lora_ckpt")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--max_train_steps", type=int, default=1000000)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--offload", type=bool, default=True)
    args = parser.parse_args()
    return args

def infer_vlm_batch(input_img_paths_batch, input_instructions_batch, prefix, vlm_model):
    outputs = []
    for input_img_path, input_instruction in zip(input_img_paths_batch, input_instructions_batch):
        
        tp = []
        for path in input_img_path:
            tp.append({"type": "image", "image": path})
        # 如果 input_instruction 是 list，需要 join
        if isinstance(input_instruction, list):
            instruction_str = " ".join(input_instruction)
        else:
            instruction_str = str(input_instruction)
        tp.append({"type": "text", "text": instruction_str + prefix})

        messages = [{"role": "user", "content": tp}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(device)

        generated_ids = vlm_model.generate(**inputs, do_sample=False, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        outputs.append(output_text[0])

    return outputs

def extract_gen_content(text):
    text = text[6:-7]
    return text

def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents

def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def no_wrap_embedding(module, recurse, nonwrapped_numel):
    # 不包裹 nn.Embedding
    if isinstance(module, nn.Embedding):
        return False
    
    # 对大模块继续用 size_based 自动包裹
    return size_based_auto_wrap_policy(module, recurse, nonwrapped_numel)

ARGS = parse_args()
logging_dir = os.path.join(ARGS.output_dir, ARGS.logging_dir)
accelerator_project_config = ProjectConfiguration(project_dir=ARGS.output_dir, logging_dir=logging_dir)

# 初始化分布式环境
accelerator = Accelerator(
        gradient_accumulation_steps=ARGS.gradient_accumulation_steps,
        mixed_precision=ARGS.mixed_precision,
        log_with=ARGS.report_to,
        project_config=accelerator_project_config,
    )
device = accelerator.device

weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

if accelerator.is_main_process:
    if ARGS.output_dir is not None:
        os.makedirs(ARGS.output_dir, exist_ok=True)

# 加载预训练模型
# vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(ARGS.vlm_path, torch_dtype="bfloat16")
# vlm_model.eval()
# vlm_model.requires_grad_(False)
# processor = AutoProcessor.from_pretrained(ARGS.vlm_path)

base_model = ARGS.base_model_path
dit = FluxTransformer2DModel.from_pretrained(base_model, subfolder = "transformer", torch_dtype=weight_dtype)
vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
t5 = T5EncoderModel.from_pretrained(base_model, subfolder="text_encoder_2", torch_dtype=weight_dtype)
clip = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=weight_dtype)
t5_tokenizer = T5Tokenizer.from_pretrained(base_model, subfolder = "tokenizer_2")
clip_tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder = "tokenizer")

vae.requires_grad_(False)
t5.requires_grad_(False)
clip.requires_grad_(False)

# 注入 LoRA 层
lora_rank = ARGS.lora_rank
lora_alpha = ARGS.lora_alpha
lora_config = LoraConfig(
    r=lora_rank, lora_alpha=lora_alpha, target_modules=["to_q", "to_v"], lora_dropout=0.05
)
dit.add_adapter(lora_config, adapter_name="edit")

# 检查可训练参数
trainable_params = [p for n, p in dit.named_parameters() if p.requires_grad]
trainable_named_params = [n for n, p in dit.named_parameters() if p.requires_grad]
# logger.info(f"trainable_named_params: {trainable_named_params}")

# 设置优化器
lr = ARGS.lr
optimizer = AdamW(trainable_params, lr=lr)

# 加载数据 
dataset = Replace5kDataset(json_file="/home/yanzhang/datasets/replace-5k/train.json")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

num_channels_latents = dit.config.in_channels // 4
dit, optimizer, dataloader = accelerator.prepare(
    dit, optimizer, dataloader
)
generator = torch.Generator(device=device).manual_seed(42)

# if accelerator.is_main_process:
#     accelerator.init_trackers(args.tracker_project_name, {"test": None})    

# 训练循环
logger.info("***** Running training *****")
# progress_bar = tqdm(
#         range(0, ARGS.max_train_steps),
#         initial=0,
#         desc="Steps",
#         disable=not accelerator.is_local_main_process,
#     )
num_epochs = ARGS.num_epochs
global_step = 0
save_steps = ARGS.save_steps
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(dit):
            instructions = batch["prompt"]
            src_image_paths = batch["src_image_path"]
            ref_image_paths = batch["ref_image_path"]
            tgt_image_paths = batch["tgt_image_path"]
            src_images = batch["src_image"].to(device)
            ref_images = batch["ref_image"].to(device)
            tgt_images = batch["tgt_image"].to(device)
            batch_size = src_images.shape[0]

            with torch.no_grad():
            # 获取vlm prompt
            # prefix = " It is editing task."
            # prompts = infer_vlm_batch([[a,b] for a,b in zip(src_image_paths, ref_image_paths)], instructions, prefix, self.vlm)
            # prompts = [extract_gen_content(prompt) for prompt in prompts]
                logger.info("************model_forward***********")
                prompts = instructions
            
                # 获取文本embedding & tokens
                clip.to(device)
                pooled_prompt_embeds = _encode_prompt_with_clip(
                    text_encoder=clip,
                    tokenizer=clip_tokenizer,
                    prompt=prompts,
                    device=device if device is not None else clip.device,
                ).to(device)
                if ARGS.offload:
                    clip.to("cpu")

                t5.to(device)
                prompt_embeds = _encode_prompt_with_t5(
                    text_encoder=t5,
                    tokenizer=t5_tokenizer,
                    prompt=prompts,
                    device=device if device is not None else t5.device,
                ).to(device)
                text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=t5.dtype)
                if ARGS.offload:
                    t5.to("cpu")
                # image preprocess
                # replace5k dataset has been preprocess during getitem
                
                # image encode
                vae.to(device)
                height = 1024
                width = 1024
                src_image_latents = (vae.encode(src_images).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
                ref_image_latents = (vae.encode(ref_images).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
                tgt_image_latents = (vae.encode(tgt_images).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
                image_latent_height, image_latent_width = src_image_latents.shape[2:]
                src_image_latents = _pack_latents(
                    src_image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                )
                ref_image_latents = _pack_latents(
                    ref_image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                )
                tgt_image_latents = _pack_latents(
                    tgt_image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                )
                src_image_ids = _prepare_latent_image_ids(
                    batch_size, image_latent_height // 2, image_latent_width // 2, device, prompt_embeds.dtype
                )
                ref_image_ids = _prepare_latent_image_ids(
                    batch_size, image_latent_height // 2, image_latent_width // 2, device, prompt_embeds.dtype
                )
                tgt_image_ids = _prepare_latent_image_ids(
                    batch_size, image_latent_height // 2, image_latent_width // 2, device, prompt_embeds.dtype
                )
                w_offset = image_latent_width // 2
                src_image_ids[..., 0] += 1
                src_image_ids[..., 2] += w_offset
                w_offset += image_latent_width // 2
                ref_image_ids[..., 0] += 2
                ref_image_ids[..., 2] += w_offset
                if ARGS.offload:
                    vae.to("cpu")
                
                # timestep
                t = torch.rand(batch_size, 1, 1, device=device) 

                # sample & add_noise
                x_1 = torch.randn_like(tgt_image_latents).to(device)
                x_t = (1 - t) * tgt_image_latents + t * x_1

            # denoise
            latent_model_input = torch.cat([x_t, src_image_latents, ref_image_latents], dim=1)
            latent_ids = torch.cat([tgt_image_ids, src_image_ids, ref_image_ids], dim=0)
            guidance = torch.full((x_t.shape[0],), 1, device=x_t.device)
            noise_pred = dit(
                    hidden_states=latent_model_input.to(dtype=weight_dtype),
                    timestep=t.squeeze(1).squeeze(1).to(dtype=weight_dtype),
                    guidance=guidance.to(dtype=weight_dtype),
                    pooled_projections=pooled_prompt_embeds.to(dtype=weight_dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
                    txt_ids=text_ids.to(dtype=weight_dtype),
                    img_ids=latent_ids.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]
            noise_pred = noise_pred[:, : x_t.size(1)]

            # loss
            diff_loss = torch.nn.functional.mse_loss(noise_pred.float(), (x_1 - tgt_image_latents).float(), reduction="mean")
            loss = diff_loss 

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        
        if accelerator.sync_gradients:
            # progress_bar.update(1)
            global_step += 1
            accelerator.log({"loss": loss.item()}, step=global_step)
            logger.info(f"loss:{loss.item()}")
            # accelerator.log({"diff_loss": diff_loss.item()}, step=global_step)
            # accelerator.log({"dino_loss": dino_loss.item()}, step=global_step)

            if global_step % ARGS.save_steps == 0:
                if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if ARGS.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(ARGS.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(ARGS.output_dir, f"checkpoint-{global_step}")

                    accelerator.save_state(save_path)
                    unwrapped_model_state = accelerator.unwrap_model(dit).state_dict()

                    # save checkpoint in safetensors format
                    lora_state_dict = {k:unwrapped_model_state[k] for k in unwrapped_model_state.keys() if '_lora' in k}
                    save_file(
                        lora_state_dict,
                        os.path.join(save_path, "lora.safetensors")
                    )

                    logger.info(f"Saved state to {save_path}")
            
        # logs = {"loss": loss.detach().item(), "diff_loss": diff_loss.detach().item(),}
        # progress_bar.set_postfix(**logs)

        if global_step >= ARGS.max_train_steps:
            break

accelerator.wait_for_everyone()
accelerator.end_training()


