# method= ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_cok"]

python3 run.py \
--dataset=aqua \
--dataset_path=./tasks/AQuA/dataset/test.json \
--prompt_path=./tasks/AQuA/prompt \
--method=few_shot_cok \
--model=gpt3-xl \
--limit_dataset_size=10 \
--max_length=256 \