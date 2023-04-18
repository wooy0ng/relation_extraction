import torch

from transformers import AutoTokenizer
from torch.utils.data import Dataset
from .utils import load_data, label_to_num


class RelationExtractionDataset(Dataset):
    def __init__(self, model_name, stage='train'):
        # load dataset
        if stage == 'train':
            self.dataset = load_data('../dataset/train/train.csv')
        else:
            self.dataset = load_data('../dataset/test/test_data.csv')

        # tokenizing
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
            
        )
        self.tokenized_dataset = self.tokenizing()
        
        # label to num        
        self.labels = label_to_num(self.dataset['label'].values)
    
    def tokenizing(self):
        concat_entity = []
        for e1, e2 in zip(self.dataset['subject_entity'], self.dataset['object_entity']):
            tmp = e1.strip() + '[SEP]' + e2.strip()
            concat_entity.append(tmp)
            
        tokenized_sentences = self.tokenizer(
            concat_entity,
            list(self.dataset['sentence']),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True
        )
        return tokenized_sentences
    
    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() for key, val in self.tokenized_dataset.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

