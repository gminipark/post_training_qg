import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random
import torch

#Random Seed
random_seed = 1
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)

class KoBARTQGDataset(Dataset):
    def __init__(self, file, tokenizer, max_len = 512, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file, sep='\t', encoding='utf-8')
        self.len = self.docs.shape[0]
        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index
        self.mask_index = self.tokenizer.mask_token_id

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def text_infiling(self, inputs):
        token_length = len(inputs)
        mask_length = inputs.count(self.mask_index)

        while (mask_length / token_length) < 0.3:
            poisson_num = np.random.poisson(3, 1).item()

            if poisson_num == 0:
                index = int(np.random.uniform(low=1, high=len(inputs) - 2))
                inputs.insert(index, self.mask_index)
            else:
                start_index = int(np.random.uniform(low=1, high=len(inputs) - 2 - poisson_num))
                inputs[start_index] = self.mask_index

                for i in range(start_index + 1, start_index + poisson_num):
                    del inputs[start_index + 1]

            mask_length += poisson_num

        return inputs

    def sentence_permutation(inputs):
        sentence_list = [sentence.strip() + '.' for sentence in inputs.split('.') if sentence != '']
        random.shuffle(sentence_list)
        inputs = ' '.join(sentence_list)
    
        return inputs

    def preprocessing_nnp(self, inputs, idx):

        nnp_words = self.nnp_words_list[idx]

        for nnp_word in nnp_words:
            current_pos = 0
            word_pos = inputs.find(nnp_word, current_pos)
            while(word_pos != -1):
                if inputs[word_pos-1] == ' ':
                    inputs = inputs[:word_pos] + '<hl1> ' + inputs[word_pos:word_pos + len(nnp_word)] + ' <hl2>' + inputs[word_pos + len(nnp_word)::]
                current_pos = word_pos + len(nnp_word) + 6
                word_pos = inputs.find(nnp_word, current_pos)

        return inputs

    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]

        content = instance['content'].strip()
        answer = instance['answer'].strip()

        content_id = self.tokenizer.encode(content)
        answer_id = self.tokenizer.encode(answer)

        if (len(content_id) + len(answer_id) + 1) < self.max_len:
            input_ids = content_id + [self.tokenizer.sep_token_id] + answer_id
        else:
            input_ids = content_id[:self.max_len - len(answer_id) - 1] + [self.tokenizer.sep_token_id] + answer_id
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(instance['question'])
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.tokenizer.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}

    def __len__(self):
        return self.len
