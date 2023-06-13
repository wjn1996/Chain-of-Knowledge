# method= ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_cok"]

python3 run.py \
--dataset=sports \
--dataset_path=./tasks/Sports/dataset/task.json \
--prompt_path=./tasks/Sports/prompt \
--method=zero_shot_cot \
--model=gpt3-xl \
--limit_dataset_size=20 \
--max_length=256 \