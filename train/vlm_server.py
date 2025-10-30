import os, argparse, json, zmq, torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from utils.vprocess import process_vision_info
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,ShardingStrategy, MixedPrecision
# 启动前设置 CUDA_VISIBLE_DEVICES=0

def build_vlm(vlm_path):
    proc = AutoProcessor.from_pretrained(vlm_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_path, torch_dtype=torch.bfloat16
    ).eval()
    return model, proc

def optimize_prompts(model, proc, items):
    prefix = " It is editing task."
    outputs = []
    for it in items:
        tp = [{"type":"image","image":p} for p in it["images"]]
        tp.append({"type":"text","text":it["text"] + prefix})
        messages = [{"role":"user","content":tp}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        
        inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out_ids = model.generate(**inputs, do_sample=False, max_new_tokens=4096)
        trimmed = [o[len(i):] for i,o in zip(inputs.input_ids, out_ids)]
        output_text = proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs.append(output_text[0])
    return outputs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm_path", required=True)
    ap.add_argument("--bind", default="tcp://0.0.0.0:5555")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)

    model, proc = build_vlm(args.vlm_path)

    print(f"[VLM] listening {args.bind}")
    while True:
        req = json.loads(sock.recv().decode("utf-8"))
        items = req["items"]
        try:
            prompts = optimize_prompts(model, proc, items)
        except Exception as e:
            print("[VLM] error:", e)
            prompts = [it["text"] for it in items]
        sock.send(json.dumps({"prompts":prompts}).encode("utf-8"))

if __name__ == "__main__":
    main()