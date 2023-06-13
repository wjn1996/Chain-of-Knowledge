# method= ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_cok"]

python3 run.py \
--dataset=last_letters \
--dataset_path=./tasks/Letter/dataset/last_letters.json \
--prompt_path=./tasks/Letter/prompt \
--method=few_shot_cok \
--model=gpt3-xl \
--limit_dataset_size=30 \
--max_length=256 \