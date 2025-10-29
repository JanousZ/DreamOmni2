import torch
import torch.distributed as dist
from torch.optim import AdamW
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig
import torch.distributed.fsdp
import torch.distributed.fsdp.api
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import argparse
from dreamomni2.pipeline_dreamomni2 import DreamOmni2Pipeline
from utils.vprocess import process_vision_info
import numpy as np
from my_datasets.replace5k import Replace5kDataset
from torch.utils.data import DataLoader
from accelerate.utils import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import ShardingStrategy
import os
from accelerate.utils import gather_object

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
    parser.add_argument("--save_dir", type=str, default="./lora_ckpt")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    return args

def infer_vlm_batch(input_img_paths_batch, input_instructions_batch, prefix):
    outputs = []
    for input_img_path, input_instruction in zip(input_img_paths_batch, input_instructions_batch):
        print(f"local_rank:{local_rank}--------------------------")
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

# 初始化分布式环境 & 日志
accelerator = Accelerator(log_with="tensorboard")

# device = accelerator.device
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")
writer = accelerator.get_tracker("tensorboard", unwrap=True)

# 加载预训练模型
ARGS = parse_args()
vlm_path = ARGS.vlm_path
base_model = ARGS.base_model_path
# vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     vlm_path, torch_dtype="bfloat16"
# )
# vlm_model.eval()
# vlm_model.to(device)

for name, param in vlm_model.named_parameters():
    param.requires_grad = False
processor = AutoProcessor.from_pretrained(vlm_path)

pipe = DreamOmni2Pipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="balanced")
pipe.vae.eval()
pipe.transformer.train()
pipe.text_encoder_2.eval()
pipe.text_encoder.eval()

# 注入 LoRA 层
lora_rank = ARGS.lora_rank
lora_alpha = ARGS.lora_alpha
lora_config = LoraConfig(
    r=lora_rank, lora_alpha=lora_alpha, target_modules=["to_q", "to_v"], lora_dropout=0.05
)
pipe.transformer.add_adapter(lora_config, adapter_name="edit")

# 检查可训练参数
trainable_params = [p for n, p in pipe.transformer.named_parameters() if p.requires_grad]
trainable_named_params = [n for n, p in pipe.transformer.named_parameters() if p.requires_grad]
# print(trainable_named_params)

# 设置优化器
lr = ARGS.lr
optimizer = AdamW(trainable_params, lr=lr)

# 加载数据 
dataset = Replace5kDataset(json_file="/home/yanzhang/datasets/replace-5k/train.json")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
pipe, optimizer, dataloader = accelerator.prepare(
    pipe, optimizer, dataloader
)
generator = torch.Generator(device=device).manual_seed(42)
# 训练循环
num_epochs = ARGS.num_epochs
global_step = 0
save_steps = ARGS.save_steps
for epoch in range(num_epochs):
    for batch in dataloader:
        with accelerator.accumulate(pipe):
            with torch.no_grad():
                # 获取vlm prompt
                instructions = batch["prompt"]
                src_image_paths = batch["src_image_path"]
                ref_image_paths = batch["ref_image_path"]
                tgt_image_paths = batch["tgt_image_path"]
                src_images = batch["src_image"]
                ref_images = batch["ref_image"]
                tgt_images = batch["tgt_image"]
                batch_size = src_images.shape[0]
                prefix = " It is editing task."

                # prompts = infer_vlm_batch([[a,b] for a,b in zip(src_image_paths, ref_image_paths)], instructions, prefix)
                # prompts = [extract_gen_content(prompt) for prompt in prompts]
                prompts = instructions
            
                # prompt encode
                (
                    prompt_embeds,
                    pooled_prompt_embeds,
                    text_ids,
                ) = pipe.encode_prompt(
                    prompt=prompts,
                    prompt_2=None,
                    prompt_embeds=None,
                    pooled_prompt_embeds=None,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=512,
                    lora_scale=None,
                )
    
                # image preprocess
                # replace5k dataset has been preprocess during getitem
                
                # image encode
                height = 1024
                width = 1024
                num_channels_latents = pipe.transformer.config.in_channels // 4
                src_image_latents = pipe._encode_vae_image(image = src_images, generator = generator)
                ref_image_latents = pipe._encode_vae_image(image = ref_images, generator = generator)
                tgt_image_latents = pipe._encode_vae_image(image = tgt_images, generator = generator)
                image_latent_height, image_latent_width = src_image_latents.shape[2:]
                src_image_latents = pipe._pack_latents(
                    src_image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                )
                ref_image_latents = pipe._pack_latents(
                    ref_image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                )
                tgt_image_latents = pipe._pack_latents(
                    tgt_image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                )
                src_image_ids = pipe._prepare_latent_image_ids(
                    batch_size, image_latent_height // 2, image_latent_width // 2, device, prompt_embeds.dtype
                )
                ref_image_ids = pipe._prepare_latent_image_ids(
                    batch_size, image_latent_height // 2, image_latent_width // 2, device, prompt_embeds.dtype
                )
                tgt_image_ids = pipe._prepare_latent_image_ids(
                    batch_size, image_latent_height // 2, image_latent_width // 2, device, prompt_embeds.dtype
                )
                w_offset = image_latent_width // 2
                src_image_ids[..., 0] += 1
                src_image_ids[..., 2] += w_offset
                w_offset += image_latent_width // 2
                ref_image_ids[..., 0] += 2
                ref_image_ids[..., 2] += w_offset
                
                # timestep
                t = torch.rand(batch_size, 1, 1, 1, device=device) 

                # sample & add_noise
                x_1 = torch.randn_like(tgt_image_latents)
                x_t = (1 - t) * tgt_image_latents + t * x_1

            # denoise
            latent_model_input = torch.cat([x_t, src_image_latents, ref_image_latents], dim=1)
            latent_ids = torch.cat([tgt_image_ids, src_image_ids, ref_image_ids], dim=1)
            noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=t,
                    guidance=None,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
            noise_pred = noise_pred[:, : x_t.size(1)]

            # loss
            diff_loss = torch.nn.functional.mse_loss(noise_pred, x_1 - tgt_image_latents)
            pred_x0 = x_t - t * noise_pred
            lambda_dino = 0
            dino_loss = 0
            loss = diff_loss + lambda_dino * dino_loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1

        if accelerator.is_main_process:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/diff_loss", diff_loss.item(), global_step)
            writer.add_scalar("train/dino_loss", dino_loss.item(), global_step)

        # 仅主进程打印日志
        if accelerator.is_main_process and global_step % 10 == 0:
            accelerator.print(f"Epoch {epoch}, Step {global_step}: loss={loss.item():.4f}")

        # 每 save_steps 步保存一次 LoRA
        if accelerator.is_main_process and global_step % save_steps == 0:
            save_dir = f"./lora_ckpt/step_{global_step}"
            save_fsdp_lora(pipe, save_dir, is_main_process=True)
            accelerator.print(f"💾 Saved LoRA checkpoint at step {global_step} -> {save_dir}")

# 训练完成后保存最终模型
if accelerator.is_main_process:
    final_dir = "./lora_ckpt/final"
    save_fsdp_lora(pipe, final_dir, is_main_process=True)
    accelerator.print(f"✅ Final LoRA weights saved to {final_dir}")

if accelerator.is_main_process:
    writer.close()
