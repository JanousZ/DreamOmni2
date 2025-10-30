export CUDA_VISIBLE_DEVICES=1 
python -m train.vlm_server --vlm_path /home/yanzhang/241/models/DreamOmni2/vlm-model &

export CUDA_VISIBLE_DEVICES=2 
python -m train.te_server --base_model_path /home/yanzhang/241/models/FLUX.1-Kontext-dev &

accelerate launch --config_file train/fsdp.yaml -m train.train