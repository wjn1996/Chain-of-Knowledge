# method= ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_cok"]

python3 run.py \
--dataset=strategyqa \
--dataset_path=./tasks/StrategyQA/dataset/task.json \
--prompt_path=./tasks/StrategyQA/prompt \
--method=few_shot_cok \
--model=gpt3-xl \
--limit_dataset_size=100 \
--max_length=256 \