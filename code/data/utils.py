import pandas as pd
import numpy as np
import pickle as pkl

from os import PathLike
from typing import Union, List


def preprocessing_dataset(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경시켜준다.
    """
    subject_entity = []
    object_entity = []
    for i, j in zip(df['subject_entity'], df['object_entity']):
        i = i[1:-1].split(",")[0].split(":")[1]     # "{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}"
        j = j[1:-1].split(",")[0].split(":")[1]     # "{'word': '조지 해리슨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'}"
    
        subject_entity.append(i)
        object_entity.append(j)
        
    out_dataset = pd.DataFrame({
        'id': df['id'],
        'sentence': df['sentence'],
        'subject_entity': subject_entity,
        'object_entity': object_entity,
        'label': df['label']
    })
    return out_dataset


def load_data(dataset_dir: Union[str, PathLike]) -> pd.DataFrame:
    df = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(df)
    
    return dataset


def num_to_label(label: np.ndarray) -> List:
    origin_label = []
    with open('data/dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pkl.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label



def label_to_num(label: np.ndarray) -> List:
    num_label = []
    with open("data/dict_label_to_num.pkl", 'rb') as f:
        dict_label_to_num = pkl.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
    return num_label