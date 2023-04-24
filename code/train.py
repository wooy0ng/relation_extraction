import torch
import wandb
import argparse
import mlflow
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datetime import datetime, timezone, timedelta
from data.dataset import RelationExtractionDataset
from model.metrics import compute_metrics


def train(args: argparse.Namespace) -> None:
    def wandb_init(project="relation extraction", name="name"):
        with open(".wandb_key", "r+") as f:
            key = f.read()

        try:
            wandb.login(key=key)
        except:
            anony = "must"
            print(
                "If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize"
            )

        time_serial = datetime.now(timezone(timedelta(hours=9))).strftime(
            "%y%m%d-%h%M%S"
        )
        wandb.init(project=project, name=f"{name}_{time_serial}")

    # load from transformers
    model_name = "klue/bert-base"

    # load dataset
    train_datamodule = RelationExtractionDataset(
        model_name=model_name, 
        stage="train",
        marker_type=args.marker_type
    )

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = 30
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_config
    ).to(device)

    # wandb 
    if args.report_to == 'wandb':
        wandb_init(name="run1")

    # init training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        warmup_steps=1000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        fp16=True,
        report_to=args.report_to if args.report_to else 'all',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_datamodule,
        eval_dataset=train_datamodule,
        compute_metrics=compute_metrics,
    )
    
    # train
    trainer.train()
    
    if args.report_to == 'mlflow':
        time_serial = datetime.now(timezone(timedelta(hours=9))).strftime("%y%m%d-%h%M%S")
        model_path = f'models/run_{time_serial}.pt'
        mlflow.pytorch.log_model(model, model_path)
    else:
        model.save_pretrained("./best_model")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser([])
    parser.add_argument('--report_to', 
        default='mlflow', 
        help=r'''input logger (wandb | mlflow | all) if not used, insert 'all'. '''
    )
    parser.add_argument('--marker_type', 
        default='entity_marker_punct', 
        help=r'''input marker type'''
    )
    
    args = parser.parse_args()
    
    train(args)
