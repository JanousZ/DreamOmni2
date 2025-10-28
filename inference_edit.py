import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    import importlib
    import transformers.utils
    import transformers.models
    origin_utils = transformers.utils
    origin_models = transformers.models
    import flash_attn
    flash_attn.hack_transformers_flash_attn_2_available_check()
    importlib.reload(transformers.utils)
    importlib.reload(transformers.models)
    origin_func = torch.nn.functional.interpolate
    def new_func(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
        if mode == "bilinear":
            dtype = input.dtype
            res = origin_func(input.to(torch.bfloat16), size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
            return res.to(dtype)
        else:
            return origin_func(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
    torch.nn.functional.interpolate = new_func
    from utils import patch_npu_record_stream
    from utils import patch_npu_diffusers_get_1d_rotary_pos_embed
    patch_npu_record_stream()
    patch_npu_diffusers_get_1d_rotary_pos_embed()
    USE_NPU = True
except:
    USE_NPU = False
from dreamomni2.pipeline_dreamomni2 import DreamOmni2Pipeline
from diffusers.utils import load_image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
from utils.vprocess import process_vision_info, resizeinput
import os
import argparse
from tqdm import tqdm
import json
from PIL import Image
import re
import argparse 

if USE_NPU:
    device = "npu"
else:
    device = "cuda"

def extract_gen_content(text):
    text = text[6:-7]
    
    return text

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
        "--input_img_path", 
        type=str, 
        nargs='+',  # Accept one or more input paths
        default=["example_input/edit_tests/src.jpg", "example_input/edit_tests/ref.jpg"], 
        help="List of input image paths (e.g., src and ref images)."
    )
    # Argument for the input instruction
    parser.add_argument(
        "--input_instruction", 
        type=str, 
        default="Make the woman from the second image stand on the road in the first image.", 
        help="Instruction for image editing."
    )
    # Argument for the output image path
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="example_input/edit_tests/edi_res.png", 
        help="Path to save the output image."
    )
    
    args = parser.parse_args()
    return args

ARGS = parse_args()
vlm_path = ARGS.vlm_path
edit_lora_path = ARGS.edit_lora_path
base_model = ARGS.base_model_path
pipe = DreamOmni2Pipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="balanced")
# pipe.to(device)

pipe.load_lora_weights(
    edit_lora_path,
    adapter_name="edit" 
)
pipe.set_adapters(["edit"], adapter_weights=[1])   


vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    vlm_path, torch_dtype="bfloat16", device_map="auto"
)
processor = AutoProcessor.from_pretrained(vlm_path)

def infer_vlm(input_img_path,input_instruction,prefix):
    tp=[]
    for path in input_img_path:
        tp.append({"type": "image", "image": path})
    tp.append({"type": "text", "text": input_instruction+prefix})
    messages = [
            {
                "role": "user",
                "content": tp,
            }
        ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = vlm_model.generate(**inputs, do_sample=False, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def infer(source_imgs,prompt):
    image = pipe(
    images=source_imgs,
    height=source_imgs[0].height,
    width=source_imgs[0].width,
    prompt=prompt,
    num_inference_steps=8,
    guidance_scale=3.5,
    ).images[0]
    return image
    

input_img_path=ARGS.input_img_path
input_instruction=ARGS.input_instruction

prefix=" It is editing task."
source_imgs = []
for path in input_img_path:
    img = load_image(path)
    # source_imgs.append(img)
    source_imgs.append(resizeinput(img))

prompt=infer_vlm(input_img_path,input_instruction,prefix)
prompt = extract_gen_content(prompt)
image=infer(source_imgs,prompt)
output_path = ARGS.output_path
image.save(output_path)