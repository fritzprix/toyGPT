from typing import Any, List
import torch
import numpy as np

import random
from torch.nn.utils.rnn import pad_sequence

import lightning as L
from datasets import Dataset, load_dataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from transformers import PreTrainedTokenizer
from torch.utils import data
from unstructured.cleaners.core import (
    replace_unicode_quotes, clean, clean_ligatures
)
import re
from transformers import PreTrainedTokenizer
from unstructured.cleaners.core import clean

import random


def random_indices(total_elements, portion, seed=-1):
    # Calculate the number of elements to select
    number_to_select = round(total_elements * portion)

    # Generate a list of unique indices for selection
    random.seed(seed)
    selected_indices = random.sample(range(total_elements), number_to_select)

    # Calculate the not-selected indices
    all_indices = set(range(total_elements))
    not_selected_indices = list(all_indices - set(selected_indices))

    return selected_indices, not_selected_indices

class SentenceChunker:
    """
    A class responsible for chunking text into sentences and tokenizing them
    according to a specified maximum length.

    Attributes:
        tokenizer (PreTrainedTokenizer): A tokenizer from the transformers library
                                         used for tokenizing sentences.
        max_length (int): The maximum token length for a single chunk.
    """

    def _split_into_sentences(self, text):
        """
        Splits the input text into sentences.

        The text is first cleaned to standardize it (removing extra whitespaces, 
        replacing unicode quotes, and removing ligatures). Then, it is split into 
        sentences using a regular expression that looks for sentence end markers 
        (., !, ?) followed by a whitespace.

        Args:
            text (str): The text to be split into sentences.

        Returns:
            List[str]: A list of sentences extracted from the input text.
        """
        # Clean the text and split it into sentences
        clean_text = replace_unicode_quotes(clean_ligatures(clean(text, extra_whitespace=True)))
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        return [f'{sentence}' for sentence in sentences]

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length:int, max_sentence_count=None, sep_token=' ', return_failure=False) -> None:
        """
        Initializes the SentenceChunker with a tokenizer and a maximum length.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenization.
            max_length (int): The maximum token length for a single chunk.
        """
        self.max_sentence_count = max_sentence_count
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_failure = return_failure
        self.sep_token = sep_token

    def __call__(self, batch, *args: Any, **kwds: Any) -> Any:
        """
        Processes a batch of text sequences by first splitting them into sentences,
        then encoding each sentence. The sentences are then chunked according to the 
        maximum length, ensuring no chunk exceeds this limit.

        Args:
            batch: A batch of text sequences.

        Returns:
            Dict[str, List]: A dictionary with two keys, 'success' and 'failure'.
                             'success' contains chunks that are within the max_length,
                             'failure' contains chunks that exceed the max_length.
        """
        # Handle single string inputs by wrapping them in a list
        if isinstance(batch, str):
            batch = [batch]

        # Split each sequence in the batch into sentences and encode them
        batch_of_chunks = [self._split_into_sentences(seq) for seq in batch]
        batch_of_encodings = [self.tokenizer.batch_encode_plus(chunks, return_length=True, add_special_tokens=True) for chunks in batch_of_chunks]

        result = {"success": []}
        if self.return_failure:
            result.update({"failure": []})
        success_batch_bucket = []
        failure_batch_bucket = []

        # Iterate over each sequence's encodings and chunk them
        for bi, encodings in enumerate(batch_of_encodings):
            bucket = []
            tokens_total = 0

            # Process each sentence in the sequence1
            for n, token_count in enumerate(encodings["length"]):
                token_count += 2 # splitting sequence removes space between two adjacent sequence in the process, so 1 token is accounted
                # Handle sentences that exceed the max length
                if token_count > self.max_length:
                    if self.return_failure:
                        failure_batch_bucket.append({"text": batch_of_chunks[bi][n], "length": token_count})
                    if len(bucket) > 0: # something in the bucket, complete a sequence and start new sequence, because dropping the middle causes discontinuity
                        success_batch_bucket.append({"text": self.sep_token.join(bucket).strip(), "length": tokens_total})
                        bucket.clear()
                        tokens_total = 0
                    continue

                if self.max_sentence_count is not None:
                    if len(bucket) >= self.max_sentence_count:
                        # if the number of setences in the bucket reaches max. 
                        # then add the sentences into success batch
                        success_batch_bucket.append({"text":self.sep_token.join(bucket).strip(), "length": tokens_total})
                        bucket.clear()
                        tokens_total = 0
                        continue

                # Check if adding the sentence would exceed the max length
                if token_count + tokens_total > self.max_length:
                    # Current bucket is full, save and reset it
                    success_batch_bucket.append({"text":self.sep_token.join(bucket).strip(), "length": tokens_total})
                    bucket.clear()
                    tokens_total = 0
                
                # Add the sentence to the current bucket
                bucket.append(batch_of_chunks[bi][n])
                tokens_total += token_count

            if len(bucket) > 0:
                success_batch_bucket.append({"text":self.sep_token.join(bucket).strip(), "length": tokens_total})
                bucket.clear()
                tokens_total = 0
            # Append the processed batches to the result
            result["success"].append([*success_batch_bucket])
            if self.return_failure:
                result["failure"].append([*failure_batch_bucket])
            success_batch_bucket.clear()
            failure_batch_bucket.clear()
        return result


def generate_choices(list_of_indices:List[int], choice_fraction):
    # Shuffle the list to ensure randomness
    k = len(list_of_indices) * choice_fraction
    if k < 1:
        return None
    list_of_indices = set(list_of_indices)

    k = int(k)
    set_of_choices = []
    while len(list_of_indices) > k:
        choices = set(random.sample(list(list_of_indices), k=k))
        list_of_indices = list_of_indices.difference(choices)
        set_of_choices.append(choices)
    if len(list_of_indices) > 0:
        set_of_choices.append(list_of_indices)
    return set_of_choices

class MLMAugmentation:

    def __init__(self,  datasets: List[Dataset], tokenizer: PreTrainedTokenizer, colunm_selection:str, sep_token_id:int, masking_fraction:float=0.15,
                 strategy:str='single') -> None:
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.column_selection = colunm_selection
        self.masking_fraction = masking_fraction
        self.sep_token_id = sep_token_id
        self.strategy = strategy


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for dataset in self.datasets:
            for data in dataset:
                for sample in data[self.column_selection]:
                    text = sample['text']
                    result = self.tokenizer(f"<cls>{text}<sep>", return_tensors="pt", return_attention_mask=False)
                    input_ids:torch.Tensor = result['input_ids']
                    poplulation = torch.nonzero(input_ids.squeeze() != self.sep_token_id).squeeze().tolist()
                    if type(poplulation) == list:
                        poplulation.remove(0)
                    else:
                        print(input_ids)
                        print(poplulation)
                        continue
                    assert 0 not in poplulation
                    choices = generate_choices(poplulation, self.masking_fraction)
                    if choices is None:
                        continue
                    label:torch.Tensor = input_ids.clone().squeeze(0)
                    if self.strategy == 'single':
                        choice = random.choice(choices[:-1])
                        input_ids[0][list(choice)] = self.tokenizer.mask_token_id
                        yield {"input": input_ids[0], "label": label}

                    elif self.strategy == 'all':
                        input_ids = input_ids.expand((len(choices), input_ids.size(-1)))
                        for i in range(len(choices)):
                            input_ids[i, list(choices[i])] = self.tokenizer.mask_token_id
                            yield {"input": input_ids[i], "label": label}
    
class CLMAugmentation:

    def __init__(self, datasets: List[Dataset], tokenizer: PreTrainedTokenizer, colunm_selection:str) -> None:
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.column_selection = colunm_selection

    def __call__(self, *args: Any, **kwds: Any) -> Any:

        for dataset in self.datasets:
            for data in dataset:
                for sample in data[self.column_selection]:
                    assert sample['length'] < 512
                    text = sample['text']
                    result = self.tokenizer(f"<|startoftext|>{text}<|endoftext|>", return_tensors="pt", return_attention_mask=False)
                    input_ids = result["input_ids"]
                    yield {"input": input_ids[0][:-1], "label": input_ids[0][1:]}
                

class MultiTaskBatchBuilder:
    def __init__(self, tokenizer, tasks):
        self.tokenizer = tokenizer
        self.tasks = tasks

    def __call__(self, data, *args, **kwargs):
        batch = {task: {'input': [], 'label': []} for task in self.tasks}
        
        # Step 1: Collect inputs and labels for each task
        for item in data:  # Iterate through each item in the data list
            for task in self.tasks:
                if task in item:
                    batch[task]['input'].append(torch.tensor(item[task]['input']))
                    batch[task]['label'].append(torch.tensor(item[task]['label']))

        # Step 2: Pad inputs and labels for each task
        for task in self.tasks:
            if batch[task]['input']:  # Check if there are any inputs to pad for the task
                batch[task]['input'] = pad_sequence(batch[task]['input'], batch_first=True, padding_value=self.tokenizer.pad_token_id)
                batch[task]['label'] = pad_sequence(batch[task]['label'], batch_first=True, padding_value=self.tokenizer.pad_token_id)

                # Step 3: Handle attention masks
                if task == 'CLM':
                    max_len = batch[task]['input'].size(1)
                    attention_masks = torch.tril(torch.ones((max_len, max_len), dtype=torch.long)).expand(batch[task]['input'].size(0), -1, -1)
                elif task == 'MLM':
                    attention_masks = (batch[task]['input'] != self.tokenizer.pad_token_id).int()
                
                batch[task]['attention_mask'] = attention_masks

        return batch

class ZippedDataset(data.Dataset):

    def __init__(self, datasets:List[data.Dataset], keys: List[str]) -> None:
        super().__init__()
        self.datasets = datasets
        self.keys = keys

    def __getitem__(self, index) -> Any:
        return {k:d for k, d in zip(self.keys, [dataset[index] for dataset in self.datasets])}

    def __len__(self):
        return np.min([len(dataset) for dataset in self.datasets])

class HFCollectionMultiTaskDataModule:

    def __init__(self, tokenizer: PreTrainedTokenizer, 
                 paths:List[str],
                 subsets: List[List[str]],
                 columns: List[str],
                 max_length:int,
                 batch_size:int,
                 tasks:List[str],
                 clear_cache:bool=False,
                 train_size:float=0.9,
                 cache_dir:str='cache',
                 num_proc=15) -> None:
        super().__init__()

        self.name = '_'.join(paths)
        self.tokenizer = tokenizer
        self.paths = paths
        self.subsets = subsets
        self.columns = columns
        self.max_length = max_length
        self.train_size = train_size
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.clear_cache = clear_cache
        self.dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.tasks = tasks
        self.cache_dir = cache_dir
        self.local_fdata_cache_path = f'{cache_dir}/{self.name}' + '_{task}/local_dscache'
        self.local_tdata_cache_path = f'{cache_dir}/{self.name}' + '_{task}/train_dscache'
        self.local_vdata_cache_path = f'{cache_dir}/{self.name}' + '_{task}/val_dscache'
        self.local_tokenized_cache_path = f'{cache_dir}/{self.name}' + '_{task}/tokenized'
        self.batch_builder = MultiTaskBatchBuilder(tokenizer=tokenizer, tasks=tasks)


    def prepare_data(self) -> None:
        for i, path in enumerate(self.paths):
            for subset in self.subsets[i]:
                dataset = load_dataset(path, subset, num_proc=self.num_proc, cache_dir=self.cache_dir)
                print(dataset)

    def setup(self) -> None:
        full_datasets = []
        train_datasets = []
        val_datasets = []
        
        datasets = [load_dataset(path, subset, num_proc=self.num_proc, cache_dir=self.cache_dir)['train'] for i, path in enumerate(self.paths) for subset in self.subsets[i]]
        for task in self.tasks:
            print(f'task : {task}')
            sentence_chunker = SentenceChunker(self.tokenizer, self.max_length - 2, sep_token=' ' if task == 'CLM' else '<sep>')
            task_specific_datasets = [dataset.map(lambda b: sentence_chunker(b[column]), batched=True, batch_size=100, num_proc=self.num_proc).flatten().select_columns(['success']) for dataset, column in zip(datasets, self.columns)]
            if task == 'CLM':
                preprocessor = CLMAugmentation(task_specific_datasets, self.tokenizer, colunm_selection="success")
            elif task == 'MLM':
                preprocessor = MLMAugmentation(task_specific_datasets, self.tokenizer, colunm_selection="success", sep_token_id=self.tokenizer.sep_token_id)
            print(f'Augmentation: {preprocessor}')
            augmented_dataset = Dataset.from_generator(preprocessor, num_proc=self.num_proc, cache_dir=self.cache_dir)
            augmented_dataset = augmented_dataset.select_columns(["input", "label"]).train_test_split(test_size=(1 - self.train_size), train_size=self.train_size)
            print(f'dataset: {augmented_dataset}')
            visible_dataset = augmented_dataset['train']
            val_selection, train_selection = random_indices(len(visible_dataset), (1 - self.train_size))
            val_dataset = visible_dataset.select(val_selection)
            print(f'val dataset : {val_dataset}')
            train_dataset = visible_dataset.select(train_selection)
            print(f'train dataset : {train_dataset}')
            full_datasets.append(augmented_dataset)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        self.full_datasets = full_datasets
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        
        if self.dataset is None:
            self.dataset = ZippedDataset(self.full_datasets, self.tasks)
        if self.val_dataset is None:
            self.val_dataset = ZippedDataset(self.val_datasets, self.tasks)
        if self.train_dataset is None:
            self.train_dataset = ZippedDataset(self.train_datasets, self.tasks)
        return len(self.train_dataset), len(self.val_dataset)
    
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = self.train_dataset
        return data.DataLoader(train_dataset,  batch_size=self.batch_size, collate_fn=self.batch_builder)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataset:Dataset = self.val_dataset
        return data.DataLoader(val_dataset, batch_size=self.batch_size, collate_fn=self.batch_builder)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_dataset = self.dataset["test"]
        return data.DataLoader(test_dataset,  batch_size=self.batch_size, collate_fn=self.batch_builder)

  