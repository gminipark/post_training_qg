# Load model directly
from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import setup_chat_format

import argparse 
import json
import natsort
import os
import torch

from datasets import load_dataset
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader


task_to_dataset = {
                  "news_summarization" : "xsum",
                  "science_summarization" : "yaolu/multi_x_science_sum",
                  "conversation_summarization" : "Samsung/samsum",
                  "question_generation" : "lmqg/qg_squad",
                  "data_to_text_generation" : "GEM/dart",
                  "commonsense_generation" : "allenai/common_gen",
    }

dataset_name_mapping = {
    "xsum": ("document", "summary"),
    "yaolu/multi_x_science_sum": ("abstract", "related_work"),
    "Samsung/samsum": ("dialogue", "summary"),
    "lmqg/qg_squad": ("paragraph_answer", "question"),
    "GEM/dart": ("tripleset", "target"),
    "allenai/common_gen": ("concepts", "target"),
}


    
    
def prepare_input(sample, model_type, system_message, text_column1_name, text_column2_name ,tokenizer):
    
    if 'llama' in model_type:
        row_json =  [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Input: {sample[text_column1_name]} \nAnswer: {sample[text_column2_name]} \nOutput: "},
            # {"role": "assistant", "content": ""}
        ]
            
        return {"messages" : tokenizer.apply_chat_template(row_json, tokenize=False)}
    
    else:
        row_json =  [
            {"role": "user", "content": f"{system_message} \n###Input: {sample[text_column1_name]} \n###Answer: {sample[text_column2_name]} \n###Question: "},
            # {"role": "assistant", "content": ""}
        ]
    
        return {"text" : tokenizer.apply_chat_template(row_json, tokenize=False)}

SYSTEM_MESSAGES = {
    "lmqg/qg_squad" : "You are an useful AI assitant. Generate a question following the Input and Answer."
}