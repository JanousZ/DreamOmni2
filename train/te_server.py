import os
import argparse
import json
import zmq
import torch
import numpy as np
from typing import Optional, Tuple

from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    T5EncoderModel,
)

# 启动前设置：CUDA_VISIBLE_DEVICES=1
# 请求：{"prompts":[...]}
# 响应：multipart，头部包含每个张量的 dtype/shape，后续为各自的 bytes

def tensor_to_bytes(t: torch.Tensor) -> Tuple[bytes, str, list]:
    arr = t.detach().contiguous().cpu().numpy()
    return arr.tobytes(), str(arr.dtype), list(arr.shape)

def load_text_encoder(
    base_model_path: str,
    subfolder_tokenizer: str = "tokenizer",
    subfolder_text_encoder: str = "text_encoder",
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
):
    tok_path = os.path.join(base_model_path, subfolder_tokenizer)
    te_path = os.path.join(base_model_path, subfolder_text_encoder)

    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)

    # 优先尝试带投影的模型
    if "2" in subfolder_tokenizer:
        text_encoder = T5EncoderModel.from_pretrained(te_path, torch_dtype=dtype)
    else:
        text_encoder = CLIPTextModel.from_pretrained(te_path, torch_dtype=dtype)

    text_encoder.eval().requires_grad_(False).to(device)
    return tokenizer, text_encoder

@torch.inference_mode()
def encode_prompts(
    tokenizer,
    text_encoder,
    prompts,
    device: torch.device,
    max_length: int = 512,
    dtype: torch.dtype = torch.bfloat16,
):
    # tokenizer 统一输出到 device
    tokens = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device, non_blocking=True)
    attn_mask = tokens.attention_mask.to(device, non_blocking=True)

    with torch.cuda.amp.autocast(dtype=dtype):
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1].to(dtype)  # [B, L, C]

        # pooled：优先 text_embeds/pooled_output，其次 EOS 位隐藏态，再次 mean pool
        pooled: Optional[torch.Tensor] = None
        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            pooled = outputs.text_embeds.to(dtype)  # [B, C] (CLIPTextModelWithProjection)
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output.to(dtype)  # [B, C]
        else:
            # 尝试 EOS 位（根据 tokenizer.eos_token_id）
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None and (input_ids == eos_id).any():
                # 取每行最后一个 eos 位置
                idx = (input_ids == eos_id).int()
                last_idx = idx.cumsum(dim=1).argmax(dim=1)  # [B]
                pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]  # [B, C]
            else:
                # 均值池化（mask）
                mask = attn_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))

    return hidden, pooled, input_ids

def maybe_load_second_encoder(base_model_path: str, device: torch.device, dtype: torch.dtype):
    tok2_path = os.path.join(base_model_path, "tokenizer_2")
    te2_path = os.path.join(base_model_path, "text_encoder_2")
    if os.path.isdir(tok2_path) and os.path.isdir(te2_path):
        tok2, te2 = load_text_encoder(
            base_model_path,
            subfolder_tokenizer="tokenizer_2",
            subfolder_text_encoder="text_encoder_2",
            device=device,
            dtype=dtype,
        )
        return tok2, te2
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_path", required=True)
    ap.add_argument("--bind", default="tcp://0.0.0.0:5556")
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    # ZeroMQ
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 主/次文本编码器（如果存在第二套就并行编码并拼接）
    tok1, te1 = load_text_encoder(args.base_model_path, "tokenizer", "text_encoder", device, dtype)
    tok2, te2 = maybe_load_second_encoder(args.base_model_path, device, dtype)

    print(f"[TE] listening {args.bind} | has_second_encoder={te2 is not None}")

    while True:
        req = json.loads(sock.recv().decode("utf-8"))
        prompts = req["prompts"]

        pe_list = []
        ppe_list = []
        tids_list = []

        # 编码器 1
        pe1, ppe1, tids1 = encode_prompts(tok1, te1, prompts, device, args.max_length, dtype)
        pe_list.append(pe1); ppe_list.append(ppe1); tids_list.append(tids1)

        # 编码器 2（可选）
        if te2 is not None:
            pe2, ppe2, tids2 = encode_prompts(tok2, te2, prompts, device, args.max_length, dtype)
            # 对于 text_ids，不同 tokenizer 可能长度不同，通常仅返回第一套
            pe_list.append(pe2); ppe_list.append(ppe2)

        # 融合策略：
        # - 最简单：concat 两个 encoder 的 embeddings/pooled（维度翻倍）
        # - 如你的 Transformer 期望拼接，请保持与训练侧一致
        if len(pe_list) == 2:
            prompt_embeds = torch.cat(pe_list, dim=-1)
            pooled_prompt_embeds = torch.cat(ppe_list, dim=-1)
            text_ids = tids_list[0]
        else:
            prompt_embeds = pe_list[0]
            pooled_prompt_embeds = ppe_list[0]
            text_ids = tids_list[0]

        fields, blobs = {}, []
        for k, v in {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
        }.items():
            b, dt, sh = tensor_to_bytes(v)
            fields[k] = {"dtype": dt, "shape": sh}
            blobs.append(b)

        sock.send_multipart([json.dumps(fields).encode("utf-8")] + blobs)

if __name__ == "__main__":
    main()