model_path="./models/train"
output_path="./outputs/train"
log_path="./logs/train"
seed=123
echo $model_path$num
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed} --lr [0.00001]*800