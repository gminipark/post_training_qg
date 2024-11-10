#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
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

"""
Pretraining the library models for T5-like span-masked language modeling on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be pretrained by this script:
https://huggingface.co/models?filter=t5
"""


from typing import Dict, List

import numpy as np
import random
from transformers import (
    BatchEncoding,
    DataCollatorForSeq2Seq
)

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

def get_preprocess_function(dataset_name):
    
    if "lmqg/qg_squad" == dataset_name:
        return preprocess_function_seq2seq
    elif "xsum" == dataset_name:
        return preprocess_function_seq2seq
    elif "Samsung/samsum" == dataset_name:
        return preprocess_function_seq2seq
    elif "GEM/dart" == dataset_name:
        return preprocess_fucntion_dart
    elif "yaolu/multi_x_science_sum" == dataset_name:
        return preprocess_fucntion_science
    elif "allenai/common_gen" == dataset_name:
        return preprocess_function_common
    else:
        raise ValueError(f"The dataset name is wrong: {dataset_name}")

def preprocess_function_seq2seq(examples, 
                                tokenizer,  
                                text_column_name, 
                                target_column_name, 
                                max_source_length, 
                                max_target_length, 
                                prefix=None,
                                padding=False):
    
    inputs, targets = [], []
    for i in range(len(examples[text_column_name])):
        if examples[text_column_name][i] and examples[target_column_name][i]:
            inputs.append(examples[text_column_name][i])
            targets.append(examples[target_column_name][i])
        else:
            print(examples[text_column_name][i])
            print(examples[target_column_name][i])

    if prefix:
        inputs = [prefix + " " + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
        
    model_inputs['labels'] = labels['input_ids']

    return model_inputs

def preprocess_fucntion_dart(examples, 
                            tokenizer,  
                            text_column_name="tripleset", 
                            target_column_name="target",
                            max_source_length=512, 
                            max_target_length=128, 
                            prefix=None,
                            padding=False):
    
    
    inputs, targets = [], []
    
    for i in range(len(examples[text_column_name])):
        if examples[text_column_name][i] and examples[target_column_name][i]:
            triple_texts = []
            for triple in examples[text_column_name][i]:
                
                head = triple[0]
                relation = triple[1]
                tail = triple[2]
                
                triple_text = f"<H> {head} <R> {relation.lower()} <T> {tail}"
                triple_texts.append(triple_text)
            
            source_text = " ".join(triple_texts)
            target = examples[target_column_name][i]
            
            inputs.append(source_text)
            targets.append(target)
                
    examples['data'] = inputs
    examples['text'] = targets
    
    
    return preprocess_function_seq2seq(examples,
                                       tokenizer, 
                                       text_column_name="data",
                                       target_column_name="text",
                                       max_source_length=max_source_length,
                                       max_target_length=max_target_length,
                                       prefix=prefix,
                                       padding=padding)
    

def preprocess_function_qg(sample, tokenizer, max_source_length, max_target_length=None, padding=False):
    prefix = "Given a paragraph and an answer, generate a question. "
    model_inputs = {}
    input_ids = []
    for idx, item in enumerate(sample['paragraph_answer']):
        
        text = prefix + " " + item
        
        text_input_ids = tokenizer(text=text, truncation=True, max_length=max_source_length, padding=padding)['input_ids']
        input_ids.append(text_input_ids)
    
    model_inputs["input_ids"] = input_ids 
    
    labels = tokenizer(text_target=sample['question'], truncation=True, max_length=max_target_length, padding=padding)['input_ids']       
        
    model_inputs['labels'] = labels

    return model_inputs


def preprocess_fucntion_science(examples, 
                            tokenizer,  
                            text_column_name='abstract', 
                            target_column_name='related_work',
                            max_source_length=512, 
                            max_target_length=128, 
                            prefix=None,
                            padding=False):
    
    
    inputs, targets = [], []
    
    for i in range(len(examples[text_column_name])):
        if examples[text_column_name][i] and examples[target_column_name][i]:
            source_texts = [examples[text_column_name][i]]
            
            metadata = examples['ref_abstract'][i]
            
            for ref_idx in range(len(metadata['cite_N'])):
                
                cite_text = " ".join([metadata['cite_N'][ref_idx], metadata['abstract'][ref_idx]])
                source_texts.append(cite_text)
            
            source_text = " ".join(source_texts)
            target = examples[target_column_name][i]
            
            inputs.append(source_text)
            targets.append(target)
                
    examples['abstracts'] = inputs
    examples['related_work'] = targets
    
    return preprocess_function_seq2seq(examples,
                                       tokenizer, 
                                       text_column_name="abstracts",
                                       target_column_name="related_work",
                                       max_source_length=max_source_length,
                                       max_target_length=max_target_length,
                                       prefix=prefix,
                                       padding=padding)

def preprocess_function_common(examples, 
                            tokenizer,  
                            text_column_name="concepts", 
                            target_column_name="target",
                            max_source_length=48, 
                            max_target_length=128, 
                            prefix=None,
                            padding=False):

    inputs, targets = [], []
    
    mask_token = tokenizer.mask_token
    for i in range(len(examples[text_column_name])):
        if examples[text_column_name][i] and examples[target_column_name][i]:
            source_text = f"{mask_token} " + f" {mask_token} ".join(examples[text_column_name][i]) + f" {mask_token}"
            target_text = examples[target_column_name][i]
            inputs.append(source_text)
            targets.append(target_text)
            
        elif examples[text_column_name][i]:
            source_text = f"{mask_token} " + f" {mask_token} ".join(examples[text_column_name][i]) + f" {mask_token}"
            target_text = " "
    
            inputs.append(source_text)
            targets.append(target_text)

    examples['concepts'] = inputs
    examples['target'] = targets
    
    
    return preprocess_function_seq2seq(examples,
                                       tokenizer, 
                                       text_column_name="concepts",
                                       target_column_name="target",
                                       max_source_length=max_source_length,
                                       max_target_length=max_target_length,
                                       prefix=prefix,
                                       padding=padding)


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded



class DataCollatorForT5Denoising(DataCollatorForSeq2Seq):
    
    def __init__(self, noise_density=0.15,
        mean_noise_span_length=3.0,
        additional_token_nums=0,
        add_eos_token=True,
        *args, **kwargs):
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.additional_token_nums = additional_token_nums
        self.add_eos_token = add_eos_token
        super(DataCollatorForT5Denoising, self).__init__(*args, **kwargs)
    
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
            
        batch = BatchEncoding(
            {k: np.array([features[i][k] for i in range(len(features))], dtype=object) for k, v in features[0].items()}
        )
        
        for idx, (context_input_ids, labels) in enumerate(zip(batch['input_ids'], batch['labels'])):
            input_ids = labels
            input_length = len(input_ids)
            
            input_ids = np.asarray([input_ids])
            
            mask_indices = np.asarray([self.random_spans_noise_mask(input_length)])
            labels_mask = ~mask_indices

            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
            
            new_input_ids = self.filter_input_ids(input_ids, input_ids_sentinel)[0]
            
            context_input_ids_length = len(context_input_ids)
            new_input_ids_length = len(new_input_ids)
            
            if  context_input_ids_length + new_input_ids_length > self.max_length:
                overflow_nums = context_input_ids_length + new_input_ids_length - self.max_length
                context_input_ids=np.concatenate([context_input_ids[:-overflow_nums-1],np.array([self.tokenizer.eos_token_id])], axis=-1)
                assert len(context_input_ids) + new_input_ids_length  == self.max_length
            
            
            new_input_ids = np.concatenate([context_input_ids,new_input_ids], axis=0).astype(dtype=np.int64)
            
            new_labels = self.filter_input_ids(input_ids, labels_sentinel)[0]
            
        
            features[idx]['input_ids'] = new_input_ids
            features[idx]['labels'] = new_labels
            
    
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        
        return features
    
    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        # start indices of corrupted spans as 1
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:,0] = mask_indices[:,0]
        # give sequential num to start index:  1, 2, ...
        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        # change start index to sentinel token num
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - self.additional_token_nums - sentinel_ids), 0)
        # add span information with mask indicices 
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        if self.add_eos_token:
            input_ids = np.concatenate(
                [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int64)], axis=-1
            )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        noise_density = self.noise_density
        
        if noise_density is None:
            noise_density = np.random.randint(1,50) / 100
        
        num_noise_tokens = int(np.round(length * noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_nonnoise_tokens = length - num_noise_tokens
        # num_noise_tokens should be less than num_noise_tokens and num_nonnoise_tokens
        num_noise_spans = int(np.round(min(num_noise_tokens, num_nonnoise_tokens) / self.mean_noise_span_length))

        
        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
                    
        num_nonnoise_spans = min(num_nonnoise_tokens, num_noise_spans)
        if np.random.choice([True, False]) and num_nonnoise_spans  + 1 <= num_nonnoise_tokens:
             num_nonnoise_spans = num_nonnoise_spans + 1

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            # segments 수에 맞춰서 스팬 갯수 만들기
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            # 시작 위치 섞기
            np.random.shuffle(mask_indices)
            # 양끝 False 추가
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            # 각 스팬의 토큰수에 맞게 인덱스 부여
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_nonnoise_spans)
        
        noise_span_lengths = np.pad(noise_span_lengths, (0,num_nonnoise_spans), constant_values=0)[:num_nonnoise_spans]
        # print(noise_span_lengths)
        
        # mix nonnoise_span과 noise_span 
        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_nonnoise_spans * 2]
        )
        
        interleaved_span_lengths = interleaved_span_lengths[:num_noise_spans + num_nonnoise_spans]
        
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        try:
            span_start_indicator[span_starts] = True
        except:
            print("num_noise_tokens: ",num_noise_tokens)
            print("num_noise_spans: ", num_noise_spans)
            print("noise_span_lengths: ",noise_span_lengths)
            print("num_nonnoise_tokens: ",num_nonnoise_tokens)
            print("num_nonnoises_spans", num_nonnoise_spans)
            print("nonnoise_span_lengths: ", nonnoise_span_lengths)
            print("length: ", orig_length)
            print("interleaved_span_lengths: ",interleaved_span_lengths)
            print("span_starts: ", span_starts)
            print("span_start_indeicator: ", span_start_indicator)
            exit()
            
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)
        
        return is_noise[:orig_length]


class DataCollatorWordDenoising(DataCollatorForSeq2Seq):
    
    def __init__(self, mask_ratio=None, *args, **kwargs):
        super(DataCollatorWordDenoising, self).__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                target = self.tokenizer.decode(feature['labels'], skip_special_tokens=True)
                # target_words = target.split(" ")
                target_words = word_tokenize(target)
                token_nums = len(target_words)
                
                mask_ratio = self.mask_ratio
                if mask_ratio is None:
                    mask_num = random.randint(1, token_nums)
                else:
                    assert type(mask_ratio) is float and mask_ratio <= 1.0
                    mask_num = int(token_nums * mask_ratio)
                    
                    if mask_num == 0 and mask_ratio > 0:
                        mask_num = 1
                        
                mask_token_indexs = sorted(random.sample(range(0,token_nums), mask_num))

                
                if "t5" in self.tokenizer.name_or_path:
                    for i, index in enumerate(mask_token_indexs[:100]):
                        target_words[index] = f"<extra_id_{i}>"
                else:
                    for i, index in enumerate(mask_token_indexs):
                        target_words[index] = self.tokenizer.mask_token
                
                    
                context = self.tokenizer.decode(feature['input_ids'], skip_special_tokens=True)
                corrupted_target = " ".join(target_words)
                
                feature.update(self.tokenizer(text=context, 
                                text_pair=corrupted_target, 
                                truncation='only_first', 
                                max_length=self.max_length))
                                                                     
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        
        return features