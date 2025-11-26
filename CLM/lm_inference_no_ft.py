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

# PREPROCESSING_FUNCTIONS = {
#     "llama": prepare_llama_input,
#     "gemma": prepare_gemma_input,
# }


class CustomCollator(object):
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        
        batch = {}
        for key in examples[0].keys():
            if key in ["messages", "text"]:
                self.tokenizer.padding_side="left" 
                batch = self.tokenizer([example[key] for example in examples],
                                                return_tensors="pt", 
                                                padding='longest', 
                                                truncation=False, 
                                                pad_to_multiple_of=8,
                                                add_special_tokens=False)
                # self.tokenizer.padding_side="right"
        return batch

def inference(args):
    
    data_cache_dir = None
    task_name = args.task_name
    dataset_name = task_to_dataset[task_name]
    dataset = load_dataset(dataset_name, cache_dir=data_cache_dir, trust_remote_code=True)

    model_name_or_path=args.base_model_name
    model_cache_dir=model_name_or_path.replace("/", "-")
    
    target_dir_path = args.output_dir
    predictions_dir_path = os.path.join(target_dir_path, "predictions")
    if not os.path.exists(predictions_dir_path):
        os.makedirs(predictions_dir_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=model_cache_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text_column_name, target_column_name = dataset_name_mapping[dataset_name]
    
    text_column2_name = ''
    if "qg_squad" in dataset_name :
        text_column2_name = 'answer'
    #     tokenizer.add_tokens("<hl>")
           
    # elif "dart" in dataset_name:
    #     tokenizer.add_tokens("<H>")
    #     tokenizer.add_tokens("<R>")
    #     tokenizer.add_tokens("<T>")

    
    # preprocess_function = PREPROCESSING_FUNCTIONS[args.model_type]
    dataset = dataset.filter(lambda example: example[text_column_name] and example[target_column_name])
    system_message = SYSTEM_MESSAGES[dataset_name]
    tokenized_dataset = dataset['test'].map(prepare_input, batched=False, num_proc=args.num_proc,
                                    load_from_cache_file = False,
                                    fn_kwargs={
                                            "model_type": args.model_type,
                                            "system_message" : system_message, 
                                            "text_column1_name" : text_column_name,
                                            "text_column2_name" : text_column2_name,
                                            "tokenizer": tokenizer,
                                        },
                                    )
    
    collator = CustomCollator(tokenizer=tokenizer)
    
    distributed_state = PartialState()   
        
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, cache_dir=model_cache_dir, torch_dtype=torch.float16, device_map=distributed_state.device
    )
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
                
        
    if 'llama' in model_name_or_path:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
       
    model.eval()
            
        
    with distributed_state.split_between_processes(tokenized_dataset) as distributed_dataset:
        
        predictions = []
        
        test_dataloader = DataLoader(
            distributed_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
    
        for step, batch in tqdm(
            enumerate(test_dataloader),
            total=len(test_dataloader),
            position=1,
            disable=not distributed_state.is_main_process,
        ):
            
            # with distributed_state.split_between_processes(batch) as batch:
            with torch.inference_mode():

                input_ids = batch['input_ids'].to(distributed_state.device) # .to(accelerator.device)
                # print("input_ids device: ",input_ids.device)
                
                outputs = model.generate(
                    input_ids,
                    # do_sample=True,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    min_new_tokens=args.min_new_tokens,
                )
                
                # distributed_state.print(tokenizer.batch_decode(input_ids))
                # distributed_state.print('---------------------')
                
                output_tokenized = [tok_out[len(tok_in):] for tok_in, tok_out in zip(batch['input_ids'], outputs)]
        
                # distributed_state.print(tokenizer.batch_decode(output_tokenized, skip_special_tokens=True))
                # exit()
                
                predictions.extend(output_tokenized)
    
        distributed_state.wait_for_everyone()
        gathered_predictions = gather_object(predictions)
        

    if distributed_state.is_main_process:
        
        decoded_predictions = tokenizer.batch_decode(gathered_predictions, skip_special_tokens=True)
        with open(os.path.join(predictions_dir_path, "predictions.txt"), "w") as f:

            lines = [prediction.strip().replace("\n","\\n") + "\n" for prediction in decoded_predictions]

            f.writelines(lines)

        # accelerator.wait_for_everyone()
        
    del model