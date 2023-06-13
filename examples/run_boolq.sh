# method= ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_cok"]

python3 run.py \
--dataset=boolq \
--dataset_path=./tasks/BoolQ/dataset/dev.jsonl \
--prompt_path=./tasks/BoolQ/prompt \
--method=zero_shot_cot \
--model=gpt3-xl \
--limit_dataset_size=20 \
--max_length=256 \