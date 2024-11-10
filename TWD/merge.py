
import torch
from transformers import (AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          HfArgumentParser,
                          Seq2SeqTrainer, 
                          Seq2SeqTrainingArguments,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
from accelerate import Accelerator
from huggingface_hub import HfFolder

import logging
import os
import json

import evaluate
import numpy as np
from dataclasses import dataclass, field
from preprocess import (get_preprocess_function,
                        DataCollatorWordDenoising,
                        DataCollatorForT5Denoising)
                        
from typing import Optional

from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType

from argparse import ArgumentParser



def main():

    parser = ArgumentParser()
    
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--lora_path", type=str)
    parser.add_argument("--output_path", type=str)


    args = parser.parse_args()
    
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    model = PeftModel.from_pretrained(model, args.lora_path)
    model = model.merge_and_unload()
    
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    
    
    
if __name__ == "__main__":
    main()