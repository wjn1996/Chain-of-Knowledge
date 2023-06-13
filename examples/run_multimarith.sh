# method= ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_cok"]

python3 run.py \
--dataset=multiarith \
--dataset_path=./tasks/MultiArith/dataset/MultiArith.json \
--prompt_path=./tasks/MultiArith/prompt \
--method=few_shot_cok \
--model=gpt3-xl \
--limit_dataset_size=10 \
--max_length=256 \