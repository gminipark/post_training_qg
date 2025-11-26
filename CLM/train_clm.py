from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          TrainingArguments,
                          BitsAndBytesConfig,
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
# from preprocess import (get_preprocess_function)
                        
from typing import Optional

# from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig, ModelConfig, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config, setup_chat_format

logger = logging.getLogger(__name__)