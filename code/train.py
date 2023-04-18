import torch
from transformers import (
    AutoConfig, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from data.dataset import RelationExtractionDataset
from model.metrics import compute_metrics

'''
TODO: refactoring
    TODO: make compute_metrics() function
    TODO: do training
        TODO: studying huggingface architecture
        TODO: using wandb
'''


def train() -> None:
    # load from transformers
    model_name = "klue/bert-base"
    
    # load dataset
    train_dataset = RelationExtractionDataset(model_name=model_name, stage='train')
    
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = 30
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_config
    ).to(device)
    
    # init training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        save_total_limit=3,
        save_steps=1000,
        num_train_epochs=3,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        fp16=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        compute_metrics=compute_metrics
    )
    
    # train
    trainer.train()
    model.save_pretrained("./best_model")
    
    return


if __name__ == "__main__":
    train()
