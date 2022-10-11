import torch
import KoBart
import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np

#Random Seed
random_seed = 1
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data file path
dev_path = 'data/test.tsv'
output_path = 'output_fine2'

# config
batch_size = 32

count = -1

model = KoBart.KoBARTConditionalGeneration().to(device)

if not os.path.exists(output_path):
    os.makedirs(output_path)

while True:
    count += 1
    # model, tokenizer init
    model.load_state_dict(torch.load(f'output_fine2/kobart_epoch_{count}.pth'))
    # model.load_state_dict(torch.load('output_fine2/kobart_best.pth'))
    model.model.eval()
    tokenizer = model.tokenizer

    # dataset
    dev_dataset = dataset.KoBARTQGDataset(dev_path, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size)

    with open(f'output_fine2/output_{count}.txt', 'w', encoding='utf-8') as f:
        for step_index, batch_data in tqdm( enumerate(dev_dataloader), f"[GENERATE]", total=len(dev_dataloader)):
    
            input_ids, decoder_input_ids, labels = tuple(value.to(device) for value in batch_data.values())

            output = model.model.generate(input_ids=input_ids, eos_token_id=tokenizer.eos_token_id, max_length=100, num_beams=5)

            for o in output:
                output = tokenizer.decode(o, skip_special_tokens=True)
                f.write(output+'\n')
