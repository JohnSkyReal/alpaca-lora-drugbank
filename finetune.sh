TOT_CUDA="0,1"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="12345"

DATA_PATH="/kaggle/input/drugbank-alpaca-no-ins/DrugBank_alpaca_no-ins.jsonl"
OUTPUT_PATH="lora-alpaca"
MODEL_PATH="decapoda-research/llama-7b-hf"
# lora_checkpoint="./lora-Vicuna/checkpoint-11600"
TEST_SIZE=0

CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--test_size $TEST_SIZE
