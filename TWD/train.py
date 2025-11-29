import torch
from transformers import (AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          HfArgumentParser,
                          Seq2SeqTrainer, 
                          Seq2SeqTrainingArguments,
                          Trainer,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
from huggingface_hub import HfFolder
from accelerate import Accelerator

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

from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType, AutoPeftModelForSeq2SeqLM
from trl import SFTTrainer, SFTConfig, ModelConfig, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config, setup_chat_format




logger = logging.getLogger(__name__)




@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    model_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_lora: bool = field(
        default=False,
        metadata={'help': "Whthter to use lora or not"}
    )
    lora_rank: int = field(
        default=32,
        metadata={'help': "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={'help': "The alpha parameter for Lora scaling"}
    )
    
    use_dora: bool = field(
        default=False,
        metadata={'help': "Whether to use dora or not"}
    )
    
    is_posttrained: bool = field(
        default=False,
        metadata={'help': "Whether the base model is post-trained or not"}   
    )
    
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    attn_implementation: str= field(
        default="sdpa",
        metadata={"help": "Attention implementation to use, e.g., default, eager, etc."}
    )


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
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
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


def main():
    
    
    accelerator = Accelerator()
    
    # parser = TrlParser((ModelArguments, DataTrainingArguments, SFTConfig))
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,Seq2SeqTrainingArguments))        
        
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    
    set_seed(training_args.seed)
    
    data_cache_dir = data_args.data_cache_dir
    task_name = data_args.task_name
    dataset_name = task_to_dataset[task_name]
    dataset = load_dataset(dataset_name, cache_dir=data_cache_dir, trust_remote_code=True)

    model_name_or_path=model_args.model_name_or_path
    
    model_cache_dir=None
    if not model_args.is_posttrained:
        model_cache_dir=model_name_or_path.replace("/", "-")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=model_cache_dir)
    
    text_column_name, target_column_name = dataset_name_mapping[dataset_name]
    max_source_length = data_args.max_source_length
    max_target_length = data_args.max_target_length
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    
    original_vocab_size = len(tokenizer)
    
    if "qg_squad" in dataset_name :
        tokenizer.add_tokens("<hl>")
        
        if "t5" in tokenizer.name_or_path:
            data_args.source_prefix = "generate question:"
            
    elif "dart" in dataset_name:
        tokenizer.add_tokens("<H>")
        tokenizer.add_tokens("<R>")
        tokenizer.add_tokens("<T>")

    preprocess_function = get_preprocess_function(dataset_name)
    dataset = dataset.filter(lambda example: example[text_column_name] and example[target_column_name])
    tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=data_args.preprocessing_num_workers,
                                    load_from_cache_file = False,
                                    fn_kwargs={"tokenizer": tokenizer, 
                                            "text_column_name" : text_column_name,
                                            "target_column_name" : target_column_name,
                                            "max_source_length": max_source_length,
                                            "max_target_length": max_target_length,
                                            "prefix" : prefix,
                                            "add_eos_token": True if 't5gemma' in tokenizer.name_or_path.lower() else False,
                                            }
                                    )
    logger.info(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")
    logger.info(tokenized_dataset)

    metric = evaluate.load("evaluate/metrics/rouge/rouge.py")

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
    lora = '-lora' if model_args.use_lora else ""
    
    if training_args.output_dir == "":
        training_args.output_dir = os.path.join(f"./{task_name}",f"{stage}{lora}", model_args.model_name_or_path.replace("/", "-"))
   
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    
    if last_checkpoint:
        
            if model_args.use_lora:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=model_cache_dir)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint, cache_dir=model_cache_dir)
            
    else:
        
        if model_args.is_posttrained:
            try:
                
                peft_config = PeftConfig.from_pretrained(model_name_or_path)
                model_cache_dir = peft_config.base_model_name_or_path.replace("/", "-")
                peft_model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=model_cache_dir)
                
                model = peft_model.merge_and_unload()
                if accelerator.is_main_process:
                    model.save_pretrained(os.path.join(training_args.output_dir, "base_model"))
                
            except:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=model_cache_dir)

        else:
        
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, 
                                                        cache_dir=model_cache_dir,
                                                        attn_implementation=model_args.attn_implementation,
                                                        # low_cpu_mem_usage=True,
                                                        )
        
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    if 't5gemma' in model_name_or_path.lower():
        model.config.decoder.vocab_size = original_vocab_size
        # print(f"model.config.decoder.vocab_size: {model.config.decoder.vocab_size}")
        # print(f"model.get_decoder().get_input_embeddings(): {model.get_decoder().get_input_embeddings()}")

        # 3. (Important) Update the config
        model.vocab_size = original_vocab_size
        
        print(f"Input embedding size: {model.get_input_embeddings()}")
        print(f"Output embedding size: {model.get_output_embeddings()}")

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_rank, 
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM, # FLAN-T5
            target_modules="all-linear", # ["q_proj","v_proj"], or "all-linear" ,
            use_dora=model_args.use_dora,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint    
    
    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    
    if data_args.post_train:
        data_collator = DataCollatorWordDenoising(
            data_args.mask_ratio,
            tokenizer,
            model=model,
            padding="max_length",
            label_pad_token_id=label_pad_token_id,
            max_length=max_source_length,
            pad_to_multiple_of=4,
        )
    else:
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding="max_length",
            label_pad_token_id=label_pad_token_id,
            max_length=max_source_length,
            pad_to_multiple_of=4
        )
    # accelerator = Accelerator()
    
   

    if data_args.post_train:
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset["validation"],
            # compute_metrics=compute_metrics,
            processing_class=tokenizer,
        )
    
    else:
        # trainer = SFTTrainer(
        #         model=model,
        #         args=training_args,
        #         data_collator=data_collator,
        #         train_dataset=tokenized_dataset['train'],
        #         eval_dataset=tokenized_dataset["validation"],
        #         # compute_metrics=compute_metrics,
        #         processing_class=tokenizer,
        #         )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset["validation"],
            # compute_metrics=compute_metrics,
            processing_class=tokenizer,
        )
        
    trainer.train(resume_from_checkpoint=checkpoint)
    
    
    with open(os.path.join(training_args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(training_args.to_dict(), f)
        
    # accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()


