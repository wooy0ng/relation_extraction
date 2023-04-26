import torch
from typing import Optional
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from torch.utils.data import Dataset
from . import marker
from .utils import load_data, label_to_num


class RelationExtractionDataset(Dataset):
    def __init__(
        self, 
        model_name: str, 
        stage: str="train", 
        marker_type: Optional[str]=None
    ):
        """
        args:
            model_name: input model's name
            stage: choice your stage (default: 'train')
                'train' : training model
                'test' : test(inference many)
                'inference' : inference one
            marker_type : choice your marker type (default: None)
                'entity_marker_punct' : use entity_marker_punct
        """
        
        self.marker_type = marker_type
        self.stage_data_exists = ['train', 'test']
        
        # load dataset
        if stage == "train":
            self.dataset = load_data("../dataset/train/train.csv")
        elif stage == "test":
            self.dataset = load_data("../dataset/test/test_data.csv")
        
        if self.marker_type == 'entity_marker_punct':
            self.marker_func = getattr(marker, self.marker_type)
        elif self.marker_type == None:
            self.marker_func = lambda e1, e2, sentence: sentence

        # tokenizing
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
        )
        
        if stage in self.stage_data_exists:
            self.tokenized_dataset = self.tokenizing(self.dataset)

            # label to num
            self.labels = label_to_num(self.dataset["label"].values)

    def tokenizing(self, dataset) -> BatchEncoding:
        concat_entities = []
        masking_sentences = []
        for e1, e2, sentence in zip(
            dataset["subject_entity"], dataset["object_entity"], dataset["sentence"]
        ):
            e1, e2 = e1.strip().replace('\'', ''), e2.strip().replace('\'', '')
            concat_entity = e1 + "[SEP]" + e2
            masking_sentence = self.marker_func(e1, e2, sentence)

            concat_entities.append(concat_entity)
            masking_sentences.append(masking_sentence)

        tokenized_sentences = self.tokenizer(
            concat_entities,
            masking_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )
        return tokenized_sentences

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach()
            for key, val in self.tokenized_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
