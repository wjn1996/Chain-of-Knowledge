# method= ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_cok"]

python3 run.py \
--dataset=csqa \
--dataset_path=./tasks/CSQA/dataset/dev_rand_split.jsonl \
--prompt_path=./tasks/CSQA/prompt \
--method=few_shot_cok \
--model=gpt3-xl \
--limit_dataset_size=20 \
--max_length=256 \