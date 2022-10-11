import torch.nn as nn
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import numpy as np
import random

#Random Seed
random_seed = 1
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)

class KoBARTConditionalGeneration(nn.Module):
    def __init__(self, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1').to(device)
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.sep_token = '<unused0>'
        self.mask_token = '<mask>'
        self.device = device

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
        self.pad_token_id = self.tokenizer.pad_token_id

        self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append('start_token')
        self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append('end_token')
        self.tokenizer.add_special_tokens({'start_token': '<hl1>'})
        self.tokenizer.add_special_tokens({'end_token': '<hl2>'})

        self.tokenizer.add_special_tokens({'mask_token': '<mask>'})
        self.tokenizer.add_special_tokens({'sep_token': '<unused0>'})
        self.model.resize_token_embeddings(len(self.tokenizer.vocab))

    def forward(self, input_ids, decoder_input_ids, labels):
        attention_mask = input_ids.ne(self.pad_token_id).float()
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id).float()

        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          decoder_input_ids=decoder_input_ids,
                          decoder_attention_mask=decoder_attention_mask,
                          labels=labels, return_dict=True)

