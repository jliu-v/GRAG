#!/bin/bash

export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

device=$1
seed=42
sample_mode=1
output_dir=output_datasets_v2

common_args="--seed ${seed}"
if [[ "${sample_mode}" == 1 ]]; then
    common_args="${common_args} \
    --sample_mode"
fi

for task in pfg empathetic convai2 spc ; do
  echo "Running task ${task} partition ${device} on device ${device}"

  task_output_dir=${output_dir}/${task}
  if [[ "${sample_mode}" == 1 ]]; then
    task_output_dir=${task_output_dir}_sample
  fi

  mkdir -p ${task_output_dir}

  log_file=${task_output_dir}/log.txt
  CUDA_VISIBLE_DEVICES=${device} JOB_SPLIT=${device} python generate_generic_convo.py \
    ${common_args} \
    --task ${task} \
    --output_dir ${task_output_dir} \
    >> ${log_file} 2>&1

done


# rm -r output_datasets/convai2/*
# mkdir -p output_datasets/convai2
#
# CUDA_VISIBLE_DEVICES=0 JOB_SPLIT=0 nohup python generate_generic_convo.py >> output_datasets/convai2/log0.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 JOB_SPLIT=1 nohup python generate_generic_convo.py >> output_datasets/convai2/log1.txt 2>&1 &


# rm -r output_datasets/convai2_v2_test/*
# mkdir -p output_datasets/convai2_v2_test

# CUDA_VISIBLE_DEVICES=1 nohup python generate_generic_convo.py --task convai2 >> output_datasets/convai2_v1/log.txt 2>&1 &

# python merge_output_data_parititions.py \
#  --partition-files output_datasets/convai2_v1/train.json \
#  --output-file output_datasets/convai2_v1/train.json \
#  --sample-output-file output_datasets/convai2_v1/sample_train.json
#
# python merge_output_data_parititions.py \
#  --partition-files output_datasets/convai2_v1/valid.json \
#  --output-file output_datasets/convai2_v1/valid.json \
#  --sample-output-file output_datasets/convai2_v1/sample_valid.json