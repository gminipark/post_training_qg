from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset

import logging
import os
import json

import evaluate
import numpy as np
from dataclasses import dataclass, field
                        
from typing import Optional

from trl import SFTTrainer, SFTConfig, ModelConfig, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    task_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    source_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    target_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the datasets downloaded from huggingface.co"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id. "
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    post_train: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to post-train the model"
            )
        },
    )
    mask_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "The probability with which to (randomly) mask tokens in the input"
            )
        },
    )

    model_type: Optional[str] = field(
        default="llama",
        metadata={
            "help": ("The type of model")
        }
    )
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )

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

def prepare_llama_input(sample, text_column_name, target_column_name, system_message, tokenizer):
    
    return {
    "messages": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": f"Input: {sample[text_column_name]} Output: "},
      {"role": "assistant", "content": sample[target_column_name]}
    ]
    }
    
def prepare_gemma_input(sample, text_column_name, target_column_name, system_message, tokenizer):
    
    row_json =  [
      {"role": "user", "content": f"{system_message} Input: {sample[text_column_name]} Output: "},
      {"role": "assistant", "content": sample[target_column_name]}
    ]
    
    return {"text" : tokenizer.apply_chat_template(row_json, tokenize=False) + tokenizer.eos_token}

SYSTEM_MESSAGES = {
    "lmqg/qg_squad" : "You are an useful AI assitant. Generate a question following the input."
}

PREPROCESSING_FUNCTIONS = {
    "llama": prepare_llama_input,
    "gemma": prepare_gemma_input,
}


def main():
    
    parser = TrlParser((ModelConfig, DataTrainingArguments, SFTConfig))
    model_config, data_args, training_args = parser.parse_args_into_dataclasses()
    
    
    # quantization_config = get_quantization_config(model_config)
    # model_kwargs = dict(
    #     revision=model_config.model_revision,
    #     trust_remote_code=model_config.trust_remote_code,
    #     attn_implementation=model_config.attn_implementation,
    #     torch_dtype=model_config.torch_dtype,
    #     use_cache=False if training_args.gradient_checkpointing else True,
    #     device_map=get_kbit_device_map() if quantization_config is not None else None,
    #     quantization_config=quantization_config,
    # )
    # training_args.model_init_kwargs = model_kwargs
    
    set_seed(training_args.seed)
    
    data_cache_dir = data_args.data_cache_dir
    task_name = data_args.task_name
    dataset_name = task_to_dataset[task_name]
    dataset = load_dataset(dataset_name, cache_dir=data_cache_dir)

    model_name_or_path=model_config.model_name_or_path
    model_cache_dir=model_name_or_path.replace("/", "-")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=model_cache_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text_column_name, target_column_name = dataset_name_mapping[dataset_name]
    # max_seq_length = training_args.max_seq_length
    
    if "qg_squad" in dataset_name :
        tokenizer.add_tokens("<hl>")
           

    metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        references = [[label] for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds, references=references) #, use_stemmer=True)
  
        return result

   
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            
    
    stage = "PT" if data_args.post_train else "FT"
    peft = '-peft' if model_config.use_peft else ""
    
    if training_args.output_dir == "":
        training_args.output_dir = os.path.join(f"./{task_name}",f"{stage}{peft}", model_config.model_name_or_path.replace("/", "-"))
   
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    
    if last_checkpoint:
        model = AutoModelForCausalLM.from_pretrained(last_checkpoint, 
            )
        
    else:
        
        quantization_config = get_quantization_config(model_config)
        
        model_kwargs = dict(
            revision=model_config.model_revision,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation=model_config.attn_implementation,
            torch_dtype=model_config.torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )

        
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                     cache_dir=model_cache_dir,
                                                     **model_kwargs
                                                     
        )
    
    
    training_args.gradient_checkpointing = training_args.gradient_checkpointing # and not model_config.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint    
    
    if "llama" in model_name_or_path:
        tokenizer.padding_side = 'right'
        # model = setup_chat_format(model)
    
     
    preprocess_function = PREPROCESSING_FUNCTIONS[data_args.model_type]
    dataset = dataset.filter(lambda example: example[text_column_name] and example[target_column_name])
    system_message = SYSTEM_MESSAGES[dataset_name]
    tokenized_dataset = dataset.map(preprocess_function, batched=False, num_proc=data_args.preprocessing_num_workers,
                                    load_from_cache_file = False,
                                    fn_kwargs={
                                            "system_message" : system_message, 
                                            "text_column_name" : text_column_name,
                                            "target_column_name" : target_column_name,
                                            "tokenizer" : tokenizer,
                                        },
                                    )
    logger.info(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")
    logger.info(tokenized_dataset)
    
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=get_peft_config(model_config),
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset["validation"],
        # compute_metrics=compute_metrics,
        processing_class=tokenizer
    )
    
    trainer.train(resume_from_checkpoint=checkpoint)
    
    
    with open(os.path.join(training_args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(training_args.to_dict(), f)

if __name__ == "__main__":
    main()

