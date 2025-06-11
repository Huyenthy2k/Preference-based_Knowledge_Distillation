#!/bin/zsh
set -e

cd ../..
echo "Current directory: $(pwd)"
cd src/prefkd/weight
echo "Current directory: $(pwd)"

model_name_1=""
model_name_2=""
# input_dir="datasets/ultra-feedback"
data_path="pvdhihihi/ultra-feedback_v2.2"
output_dir="generated-data/ultra-feedback-tisdpo"
# model1_template="normal"
# model2_template="normal"
batch_size=32
num_gpus=8
force_sequential=false  # Set to true if multiprocessing causes issues
split="train"

# Create output directory if it doesn't exist
# mkdir -p $output_dir

# Run the parallel processing script
python3 weight.py \
  --positive_model_name $model_name_1 \
  --negative_model_name $model_name_2 \
  --spilt=$split \
  --input_dir $input_dir \
  --output_dir $output_dir \
  --batch_size $batch_size \
  --num_gpus $num_gpus \
  $(if $force_sequential; then echo "--force_sequential"; fi) 
