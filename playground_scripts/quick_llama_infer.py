import os
import sys
import logging
from argparse import ArgumentParser

import pandas as pd

import transformers
import torch

logger = logging.getLogger("LLAMA_INFER")
logger.setLevel(logging.DEBUG)

PROMPT_TEMPLATE = """Given the following text {text}

Please answer this question and provide your confidence level for each target. Note that the confidence level indicates the degree of certainty you have about your answer and is represented as a percentage. 

Please provide a step by step analysis using the following format: 
Explanation: [insert step-by-step analysis here] 
Answer and Confidence (0-100): [just the answer] [just the confidence numerical number]%" """

def format_prompts(prompts):
    temp = [{"role": "system", "content": "You are a helpful bot performs text analysis"}]
    return temp + [{"role": "user", "content": prompt} for prompt in prompts]

def infer(prompts):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(device)

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
    )

    messages = format_prompts(prompts)

    outputs = pipeline(
        messages,
        max_new_tokens=512,
    )

    return outputs

def create_prompts(text):
    return PROMPT_TEMPLATE.format(text=text)


def get_prompts(file_path):
    data = pd.read_csv(file_path)

    return data["text"].apply(lambda x: create_prompts(x)).tolist()


def init_logging_config(file_path):
    console_format = logging.Formatter('[%(name)s][%(levelname)s] - %(message)s')
    file_format = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')

    console_handler = logging.StreamHandler(); console_handler.setLevel(logging.INFO); console_handler.setFormatter(console_format)
    file_handler = logging.FileHandler(file_path, mode="w"); file_handler.setLevel(logging.DEBUG); file_handler.setFormatter(file_format)

    logger.addHandler(console_handler); logger.addHandler(file_handler)


def parse_args(argv):
    argv.pop(0)

    parser = ArgumentParser()
    parser.add_argument("-lf", "--logfile_path", action="store", dest="logfile_path", required=True, type=str, help="Path for logfile")
    parser.add_argument("-dp", "--data_path", action="store", dest="data_path", required=True, type=str, help="Path to the data file")
    parser.add_argument("-of", "--output_file", action="store", dest="output_path", required=True, type=str, help="Path for the output file")

    args = parser.parse_args(argv)

    return args


if __name__ == "__main__":
    args = parse_args(sys.argv)
    init_logging_config(args.logfile_path)
    logger.info("Logging file init")

    logger.info("Creating prompts")
    prompts = get_prompts(args.data_path)
    logger.info("Prompts created")

    logger.info("Feeding prompts into the model")
    output = infer(prompts)

    with open(args.output_path, "w") as file:
        file.write(repr(output))

    