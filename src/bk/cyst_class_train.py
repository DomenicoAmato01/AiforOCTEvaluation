from utils.model import get_classification_model
import pandas as pd
from datasets.OCTDatasetForClassification import OCTDatasetForClassification
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from torch.utils.data import DataLoader
from transformers import BaseImageProcessor, BaseImageProcessorFast, EarlyStoppingCallback, Trainer, TrainingArguments
import torch
import numpy as np
from utils.model import compute_metrics, collate_fn
from utils.model import get_transform

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")


    training_args ={
        "do_train": True,
        "do_eval": True,
    }

    cyst_processor, cyst_model, model_name = get_classification_model("Cyst")
    print(cyst_model)

    cyst_model.to(device)

    df = pd.read_excel("data/esaso_eval/cyst_train_test_split.xlsx")
    
    fn = df["nomefile"].to_numpy()
    labels = df["c"].to_numpy()

    size = cyst_processor.size
    _transf = get_transform(size)


    trainset = OCTDatasetForClassification(excel_file="data/esaso_eval/cyst_train_test_split.xlsx",
                                          root_dir="data/Cyst",
                                          transform=_transf,
                                          train=True)

    testset = OCTDatasetForClassification(excel_file="data/esaso_eval/cyst_train_test_split.xlsx",
                                         root_dir="data/Cyst",
                                         transform=_transf,
                                         train=False)

    args = TrainingArguments(
        f"test-cyst-model",
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=100,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir='logs',
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=cyst_model,
        args=args,
        train_dataset=trainset if training_args["do_train"] else None,
        eval_dataset=testset if training_args["do_eval"] else None,
        compute_metrics=compute_metrics,
        processing_class=BaseImageProcessor(),
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(f"models/cyst-model/{model_name}")
    
