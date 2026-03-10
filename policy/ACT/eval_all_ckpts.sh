#!/bin/bash
task_name=$1
task_config=$2
expert_data_num=$3
seed=$4
gpu_id=$5

CKPT_DIR="./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}"

# Backup original policy_last.ckpt
cp "${CKPT_DIR}/policy_last.ckpt" "${CKPT_DIR}/policy_last.ckpt.bak"

for ckpt in policy_epoch_2000_seed_0.ckpt policy_epoch_4000_seed_0.ckpt policy_best.ckpt policy_epoch_6000_seed_0.ckpt; do
    echo "============================================"
    echo "Evaluating: ${ckpt}"
    echo "============================================"
    cp "${CKPT_DIR}/${ckpt}" "${CKPT_DIR}/policy_last.ckpt"
    bash eval.sh ${task_name} ${task_config} ${task_config} ${expert_data_num} ${seed} ${gpu_id} 2>&1
    echo ""
    echo "Done evaluating: ${ckpt}"
    echo ""
done

# Restore original
mv "${CKPT_DIR}/policy_last.ckpt.bak" "${CKPT_DIR}/policy_last.ckpt"
echo "All evaluations complete!"
