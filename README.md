# Post-Training with Interrogative Sentences for Enhancing BART-based Korean Question Generator
Post-Training for Question Generator implemented in Python. This repository is based on Post-Training with Interrogative Sentences for Enhancing BART-based Korean Question Generator from AACL-IJCNLP 2022.

# Post-training Method List
This list is based on ablation study at (Park et al, 2022.)

1. Korean Spacing(KS)
2. Korean Spacing_Augmentation(KS_AUG)
3. Question Infilling(QI)
4. Question Infilling_Augmentation(QI_AUG)
5. Question Infilling_Korean Spacing(QI_KS)
6. Question Infilling_Korean Spacing_Augmentation(QI_KS_AUG)
7. Only fine-tuning(vanilla)

# Directory
```bash
├── code
   ├── KS
   ├── KS_AUG
   ├── QI
   ├── QI_AUG
   ├── QI_KS
   ├── QI_KS_AUG
   └── vanilla
``` 

All of directory of code has KoBart.py, dataset.py, generate.py, post_dataset.py, post_train.py, train.py
 - KoBart.py: Defining model for training. It is based KoBART[https://github.com/SKT-AI/KoBART].
 - dataset.py: Utility code for loading and processing fine-tuning dataset.
 - generate.py: Inference code. It is based beam search algorithm. Default beam size is 5.
 - post_dataset.py: Utility code for loading and processing post-training dataset. objective function for post-training is implemented in this Python code.
 - post_train.py: Main code for post-training.
 - train.py: Main code for fine-tuning.

 # Dataset
 ```bash
 [Post-training]
  - Train: 243,425
  - reference: AI hub MRC dataset[https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=89]
 
 [Fine-tuning]
  - Train: 54,369
  - Validation: 6,038
  - Test: 5,574
  - reference: KorQuAD dataset[https://korquad.github.io/KorQuad%201.0/]
 ```
 
# Training and Inference

1. Post-training (Not for vanilla)
```bash
python post_train.py
```

2. Fine-tuning
```bash
python train.py
```

3. Inference
```bash
python generate.py
```

Evaluation

1. vanilla
| |BLEU-4|ROUGE-L|METEOR|
|------|:-------:|:-------:|:-------:|
|Score|20.12|38.81|34.20|
