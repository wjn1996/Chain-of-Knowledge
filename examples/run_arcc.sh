# method= ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_cok"]

python3 run.py \
--dataset=arc-c \
--dataset_path=./tasks/ARC-c/dataset/ARC-Challenge-Test.jsonl \
--prompt_path=./tasks/ARC-c/prompt \
--method=few_shot_cok \
--model=gpt3-xl \
--limit_dataset_size=20 \
--max_length=256 \