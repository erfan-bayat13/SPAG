# Self-Play of Adversarial Language Game (SPAG)

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/Linear95/SPAG/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/Linear95/SPAG/blob/main/DATA_LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

This repo contains the implementation of the paper:

- [Self-playing Adversarial Language Game Enhances LLM Reasoning](https://arxiv.org/abs/2404.10642)
    

## Environment
To build the running environment, use the following command:
```
pip3 install -r requirements.txt
```

## Imitation Learning


```bash
torchrun --nproc_per_node=8 --master_port=6000 train.py \
    --output_dir <path_to_save_your_imitation_checkpoint> \
    --model_name_or_path "Llama-2-7b-hf" \
    --ref_model_name_or_path "Llama-2-7b-hf" \
    --lm_kl_coeff 0.1 \
    --train_method "sft_weighted_with_kl" \
    --train_data_path <path_to_imitation_learning_data> \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy no \
    --padding_side "right" \
    --truncation_side "left" \
    --max_length 2048 \
    --save_strategy epoch \
    --learning_rate 5e-6 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --weight_decay 0. \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --gradient_checkpointing True \
    --tf32 True  --bf16 True
```

## Self-play of Adversarial Language Game



## Reinforcement Learning on self-play episodes



## Citation
```
@article{cheng2024SPAG,
  title={Self-playing Adversarial Language Game Enhances LLM Reasoning},
  author={Cheng, Pengyu and Hu, Tianhao and Xu, Han and Zhang, Zhisong and Dai, Yong and Han, Lei and Du, Nan},
  journal={arXiv preprint arXiv:2404.10642},
  year={2024}
}
```
