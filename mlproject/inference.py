import torch
import os
import argparse
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification
from typing import Union, DefaultDict
from .data.dataset import RelationExtractionDataset
from .data.utils import num_to_label
from omegaconf import OmegaConf

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@torch.no_grad()
def inference_one(args: Union[argparse.Namespace, DefaultDict]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_datamodule = RelationExtractionDataset(
        model_name="klue/bert-base",
        stage="inference",
        marker_type="entity_marker_punct",
    )
    if type(args) == dict:
        args = OmegaConf.create(args)
    
    df = pd.DataFrame({
        "sentence": [args.sentence],
        "subject_entity": [args.subject_entity],
        "object_entity": [args.object_entity],
    })
    tokenized_dataset = inference_datamodule.tokenizing(df).to(device)

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=f"{parent_dir}/mlproject/results/checkpoint-8118"
    ).to(device)
    model.eval()

    output = model(**tokenized_dataset)
    logits = output["logits"].detach().cpu().numpy()
    pred = np.argmax(logits, axis=-1)
    pred = num_to_label(pred)[0]

    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser([])
    parser.add_argument(
        "--sentence", default="지난 15일 MBC '탐사기획 스트레이트'가 이 사실을 보도했다.", type=str
    )
    parser.add_argument("--subject_entity", default="MBC", type=str)
    parser.add_argument("--object_entity", default="탐사기획 스트레이트", type=str)

    args = parser.parse_args()

    inference_one(args)
