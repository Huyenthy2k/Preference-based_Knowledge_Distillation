#!/bin/bash
set -e

cd ../../..
echo "Current directory: $(pwd)"
cd src/prefkd
echo "Current directory: $(pwd)"

export HYDRA_FULL_ERROR=1

/usr/local/envs/prefKD_train/bin/python -u train.py \
  model=KD_tisdpo \
  model.policy_name_or_path=tonyshelby/Qwen2.5_0.5B_SFT_sample \
  model.reference_name_or_path=Qwen/Qwen3-0.6B-Base \
  model.teacher_tokenizer_name_or_path=Qwen/Qwen3-0.6B-Base \
  model.student_tokenizer_name_or_path=tonyshelby/Qwen2.5_0.5B_SFT_sample \
  model.teacher_name_or_path=Qwen/Qwen3-0.6B-Base \
  model.student_name_or_path=tonyshelby/Qwen2.5_0.5B_SFT_sample \
  model.policy_block_name=Qwen2DecoderLayer \
  model.reference_block_name=Qwen3DecoderLayer \
  loss=KD_tisdpo \
  policy_mode=student \
  reference_mode=teacher \
  eval_every=5 \
  datasets=tonyshelby/sample \
  gradient_accumulation_steps=2 batch_size=4 eval_batch_size=4 \
  trainer=FSDPTrainer sample_during_eval=false \
  save_repo=tonyshelby/Qwen2.5_0.5B_TDPO \
#   model.policy_name_or_path=Qwen/Qwen3-0.6B-Base \
#   model.reference_name_or_path=Qwen/Qwen3-0.6B-Base \
#   model.teacher_tokenizer_name_or_path=Qwen/Qwen3-0.6B-Base \
#   model.student_tokenizer_name_or_path=Qwen/Qwen3-0.6B-Base \
#   model.policy_block_name=Qwen2DecoderLayer \
#   model.reference_block_name=Qwen2DecoderLayer \
