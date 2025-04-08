import os
import sys
import logging
from argparse import ArgumentParser

import pandas as pd

import transformers
import torch

#from llama_cpp import Llama

logger = logging.getLogger("LLAMA_INFER")
logger.setLevel(logging.DEBUG)

PROMPT_TEMPLATE = """Given the following text {text}. What are the main topics of this text?

Please answer this question and provide your confidence level for each topic. Note that the confidence level indicates the degree of certainty you have about your answer and is represented as a percentage. 

Please provide a step by step analysis using the following format: 
Explanation: [insert step-by-step analysis here] 
Conclusion - Answers and Confidences (0-100): [just the first topic] [just the confidence numerical number]%, [just the second topic] [just the confidence numerical number]%, and so on """


def log_potential_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            logger.exception("Error")
    return wrapper


def format_prompts(prompts):
    temp = [{"role": "system", "content": "You are a helpful bot performs text analysis"}]
    return temp + [{"role": "user", "content": prompt} for prompt in prompts]


@log_potential_error
def infer(prompts, model_id, hf_token):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info("device: {d}".format(d=device))

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, 
        token=hf_token,
        quantization_config=bnb_config
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, 
        token=hf_token, 
        max_length=1024
    )

    logger.info("Model loaded or downloaded!")

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        #model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False
    )

    #messages = format_prompts(prompts)
    logger.info("Prompts formated")
    
    outputs = pipeline(
        prompts,
        max_new_tokens=1024,
        batch_size=1
    )

    return outputs


def create_prompts(text):
    return PROMPT_TEMPLATE.format(text=text)


@log_potential_error
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
    parser.add_argument("-hf", "--hf_token", action="store", dest="hf_token", required=True, type=str, help="Huggingface token file")
    parser.add_argument("-m", "--model_id", action="store", dest="model_id", required=True, type=str, help="Model id or path to local model")

    args = parser.parse_args(argv)

    return args


@log_potential_error
def main():
    args = parse_args(sys.argv)
    init_logging_config(args.logfile_path)
    logger.info("Logging file init successful")

    logger.info("Creating prompts")
    prompts = get_prompts(args.data_path)
    logger.info("Prompts creation successful")

    logger.info("Feeding prompts into the model")
    with open(args.hf_token, "r") as file:
        hf_token = file.readline()
    output = infer(prompts, args.model_id, hf_token)
    logger.info("Inference successful")

    with open(args.output_path, "w") as file:
        file.write(repr(output))


if __name__ == "__main__":
    main()

    