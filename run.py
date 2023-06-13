import os
import argparse
import numpy as np
from tqdm import tqdm
import random
import json
from framework.utils import answer_cleansing, save_outputs
from framework.model import Decoder
from framework.data_processor import data_reader
from framework.prompt_engineering import PromptHub


def main(args):
    # obtain query and ground-truth label
    questions, answers = data_reader(args)
    # print("questions[0]=", questions[0])
    # print("answers[0]=", answers[0])

    # obtain prompt (manual annotated and retrieve from KBs)
    prompter = PromptHub(args=args)

    # concatenate exemplars with each query
    questions_prompts = prompter.construct_query_examples(questions)
    
    # assert 1>2
    # inference by LLM (e.g., GPT-3 text-davinci-002)
    decoder = Decoder(args=args)
    total = 0
    correct_list = []
    save_results = list()
    # factuality_score = -1
    # faithfulness_score = -1
    idx = -1
    for query, input_prompt, label in zip(questions, questions_prompts, answers):
        idx += 1
        output = decoder.decode(args, input_prompt)
        if args.method == "zero_shot_cot":
            # if zero-shot-cot, we should obtain the final answer
            input_prompt = prompter.prompting_with_output(output, query)
            output = decoder.decode(args, input_prompt)

        print("{}".format(input_prompt + output))

        pred = answer_cleansing(args, output)
        label = label.strip()
        print("pred : {}".format(pred))
        print("label : " + label)
        print('*************************')


        # save results
        save_results.append({
            "idx": idx,
            "prompt": input_prompt,
            "pred": pred,
            "label": label,
            "reasoning_chains": output,
        })
        
        # Checking answer ...
        correct = (np.array([pred]) == np.array([label])).sum().item()
        correct_list.append(correct)
        total += 1 #np.array([y]).size(0)

    # Calculate accuracy without rethinking ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))

    # saving
    save_outputs(args, save_results, 1)


    # rethinking algorithm
    if args.method == "few_shot_cok":
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chain-of-knowledge")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None, help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )
    
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    
    parser.add_argument(
        "--dataset", type=str, default="aqua", help="dataset used for experiment"
    )

    parser.add_argument(
        "--dataset_path", type=str, default="", help="dataset path"
    )

    parser.add_argument(
        "--prompt_path", type=str, default="", help="prompt path"
    )
    
    parser.add_argument(
        "--model", type=str, default="gpt3-xl", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    
    parser.add_argument(
        "--method", type=str, default="zero_shot_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_cok"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="maximum length of output tokens by model for reasoning extraction"
    )

    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/", help="save directory"
    )
    
    args = parser.parse_args()

    main(args=args)