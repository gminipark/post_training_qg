# %%
# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate

import os
from tqdm import tqdm
import numpy as np
from glob import glob
import natsort
import argparse
import json

from accelerate import Accelerator

from preprocess import get_preprocess_function
from peft import PeftModel


task_to_dataset = {
    "question_generation": "lmqg/qg_squad",
    "summarization": "xsum",
    "data_to_text_generation": "GEM/dart",
    "science_summarization": "yaolu/multi_x_science_sum",
    "conversation_summarization" : "Samsung/samsum"
}


def collate_fn(examples):

    batch = {}
    for key in examples[0].keys():
        if key == "input_ids":
            batch[key] = torch.tensor([example[key] for example in examples])
        elif key == "labels":
            batch[key] = torch.tensor([example[key] for example in examples])

    return batch


def inference(args):

    cache_dir = "./data"

    dataset_name = task_to_dataset[args.task]

    dataset = load_dataset(dataset_name, cache_dir=cache_dir)

    model_id = args.base_model_name

    cache_dir = model_id.replace("/", "-")

    prefix = None
    # Load tokenizer of FLAN-t5-base
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if "qg_squad" in dataset_name:
        tokenizer.add_tokens("<hl>")
        text_column_name = "paragraph_answer"
        target_column_name = "question"

    elif "dart" in dataset_name:
        tokenizer.add_tokens("<H>")
        tokenizer.add_tokens("<R>")
        tokenizer.add_tokens("<T>")
        text_column_name = "tripleset"
        target_column_name = "target"

    elif "xsum" in dataset_name:
        text_column_name = "document"
        target_column_name = "summary"

    elif "science" in dataset_name:
        text_column_name = "abstract"
        target_column_name = "related_work"
    elif "samsum" in dataset_name:
        text_column_name = "dialogue"
        target_column_name = "summary"

    preprocess_function = get_preprocess_function(dataset_name)

    tokenized_dataset = dataset["test"].map(
        preprocess_function,
        batched=True,
        num_proc=args.num_proc,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_source_length": args.max_source_length,
            "max_target_length": None,
            "text_column_name": text_column_name,
            "target_column_name": target_column_name,
            "padding": "max_length",
            "prefix": prefix,
        },
    )
    tokenized_dataset = tokenized_dataset.select_columns(["input_ids", "labels"])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset.features)}")
    print(tokenized_dataset)

    test_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    target_dir_path = args.checkpoint
    if args.checkpoint is not None:
        checkpoints = natsort.natsorted(
            glob(os.path.join(target_dir_path, args.checkpoint_prefix))
        )

    predictions_dir_path = os.path.join(target_dir_path, "predictions")
    if not os.path.exists(predictions_dir_path):
        os.mkdir(predictions_dir_path)

    accelerator = Accelerator()
    test_dataloader = accelerator.prepare_data_loader(
        test_dataloader, device_placement=True
    )

    metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, refers = eval_preds
        result = metric.compute(predictions=preds, references=refers)

        return result

    for idx, checkpoint in tqdm(
        enumerate(checkpoints), total=len(checkpoints), position=0
    ):

        checkpoint_dir_path = os.path.join(predictions_dir_path, f"checkpoint-{idx+1}")
        if not os.path.exists(checkpoint_dir_path):
            os.mkdir(checkpoint_dir_path)
        else:
            if os.path.exists(os.path.join(checkpoint_dir_path, "predictions.txt")):
                accelerator.free_memory()
                continue

        model_path_or_name = checkpoint
       
        
        if args.use_lora:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, cache_dir=cache_dir
            )
            embedding_size = model.get_input_embeddings().weight.shape[0]
            if len(tokenizer) > embedding_size:
                model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(model, model_path_or_name)
            model = model.merge_and_unload() 
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path_or_name, cache_dir=cache_dir
            )

        model = accelerator.prepare(model)
        model.eval()

        predictinos = []
        references = []

        samples_seen = 0
        for step, batch in tqdm(
            enumerate(test_dataloader),
            total=len(test_dataloader),
            position=1,
            disable=not accelerator.is_main_process,
        ):

            with torch.no_grad():

                input_ids = batch["input_ids"]
                labels = batch["labels"]
               
                
                outputs = accelerator.unwrap_model(model).generate(
                    input_ids,
                    max_new_tokens=args.max_target_length,
                    num_beams=args.num_beams,
                    min_new_tokens=args.min_new_tokens
                )
            # accelerator.print(len(outputs))
                outputs = accelerator.pad_across_processes(
                    outputs, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = accelerator.pad_across_processes(
                    labels, dim=1, pad_index=tokenizer.pad_token_id
                )

                outputs, labels = accelerator.gather((outputs, labels))
                # accelerator.print(len(outputs))

                decoded_preds = tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )

                if accelerator.num_processes > 1:
                    if step == len(test_dataloader) - 1:
                        decoded_preds = decoded_preds[
                            : len(test_dataloader.dataset) - samples_seen
                        ]
                        decoded_labels = decoded_labels[
                            : len(test_dataloader.dataset) - samples_seen
                        ]
                    else:
                        samples_seen += len(decoded_labels)

            predictinos.extend(decoded_preds)
            references.extend(decoded_labels)

        if accelerator.is_main_process:


            with open(os.path.join(checkpoint_dir_path, "predictions.txt"), "w") as f:

                lines = [prediction + "\n" for prediction in predictinos]

                f.writelines(lines)

            with open(os.path.join(checkpoint_dir_path, "resutls.json"),'w') as f:
                metrics = compute_metrics((predictinos, references))
       
                json.dump(metrics,f)
            
        accelerator.free_memory()
        del model

    # accelerator.clear()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "question_generation",
            "summarization",
            "data_to_text_generation",
            "science_summarization",
            "conversation_summarization"
        ],
    )

    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default="checkpoint-*/",
    )

    parser.add_argument(
        "--base_model_name",
        type=str,
        default="google/t5-v1_1-large",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
    )
    
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=0,
    )
    
    parser.add_argument(
        '--use_lora',
        action="store_true"
    )

    args = parser.parse_args()

    inference(args)


if __name__ == "__main__":
    main()
