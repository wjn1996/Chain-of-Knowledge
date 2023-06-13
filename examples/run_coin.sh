# method= ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_cok"]

python3 run.py \
--dataset=coin_flip \
--dataset_path=./tasks/Coin/dataset/coin_flip.json \
--prompt_path=./tasks/Coin/prompt \
--method=few_shot_cok \
--model=gpt3-xl \
--limit_dataset_size=10 \
--max_length=256 \