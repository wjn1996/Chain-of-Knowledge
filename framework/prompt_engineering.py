import os
import json
import argparse

TRIGGER = "The answer is"
TRIGGER2 = "So the answer is"

class PromptHub(object):
    def __init__(self, args) -> None:
        self.prompt_path = args.prompt_path
        self.method = args.method
        self.query_template = "Q:"
        self.answer_template = "A:"
        # self.answer_template_fewshot = "A: The answer is"
        # self.answer_template_cok = "A: So the answer is"
        self.prompt = self.load_prompt()
        # if len(self.prompt) > 0 and self.prompt[-1] == "\n":
        #     self.prompt = self.prompt[:-1]
    
    def load_prompt(self):
        # method= ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_cok"]
        file_name = None
        prompt = ""
        method = self.method
        if method == "few_shot":
            file_name = "prompt_standard.txt"
        elif method == "few_shot_cot":
            file_name = "prompt_cot.txt"
        elif method == "few_shot_cok":
            file_name = "prompt_cok.txt"
        
        if file_name is not None:
            with open(os.path.join(self.prompt_path, file_name), "r", encoding="utf-8") as fr:
                prompt = fr.read()
        else:
            if method == "zero_shot_cot":
                prompt = "Let's think step by step."
        
        return prompt

    def prompting(self, query):

        if self.method == "zero_shot":
            return "{} {}\n{}".format(self.query_template, query, self.answer_template)
    
        if self.method == "zero_shot_cot":
            return "{} {}\n{} {}".format(self.query_template, query, self.answer_template, self.prompt)

        if self.method == "few_shot":
            return "{}\n{} {}\n{}".format(self.prompt, self.query_template, query, self.answer_template)
        
        if self.method == "few_shot_cot":
            return "{}\n{} {}\n".format(self.prompt, self.query_template, query)
        
        if self.method == "few_shot_cok":
            return "{}\n{} {}\n".format(self.prompt, self.query_template, query)
        
        return "{} {}\n{}".format(self.query_template, query, self.answer_template)
    
    def prompting_with_output(self, output, query):
        return "{} {}\n{} {} {}".format(self.query_template, query, self.answer_template, output, TRIGGER)

    def construct_query_examples(self, questions):
        new_questions = list()
        for query in questions:
            new_questions.append(self.prompting(query))
        return new_questions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chain-of-knowledge")

    parser.add_argument(
        "--prompt_path", type=str, default="./tasks/CSQA/prompt/", help="prompt path"
    )
    
    parser.add_argument(
        "--method", type=str, default="few_shot_cok", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_cok"], help="method"
    )
    args = parser.parse_args()

    prompter = PromptHub(args)
    prompt = prompter.load_prompt()
    print("prompt=", prompt)
