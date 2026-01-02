from sklearn.metrics import ConfusionMatrixDisplay
from utils.model import get_classification_model, get_transform
from utils.model import compute_metrics, collate_fn
import pandas as pd
from datasets.OCTDatasetForClassification import OCTDatasetForClassification   
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from torch.utils.data import DataLoader
from transformers import BaseImageProcessor, BaseImageProcessorFast, Trainer, TrainingArguments, ViTForImageClassification
import numpy as np
from matplotlib import pyplot as plt
import json
import os



if __name__ == "__main__":
    
    cyst_processor, cyst_model, model_name = get_classification_model("Cyst")
    print(cyst_model)

    # cyst_model.load_pretrained("test-cyst-model")
    cyst_model = ViTForImageClassification.from_pretrained(f"models/cyst-model/{model_name}", local_files_only=True)

    size = cyst_processor.size
    _transf = get_transform(size)

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
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=cyst_model,
        args=args,
        train_dataset=None,
        eval_dataset=testset,
        compute_metrics=compute_metrics,
        processing_class=BaseImageProcessor(),
        data_collator=collate_fn,
    )

    eval = trainer.evaluate()
    print(eval)
    disp = ConfusionMatrixDisplay(np.array(eval["eval_confusion_matrix"]))

    disp.plot()
    plt.savefig(f"plot/confusion-matrix/{model_name}-cm.png")
    plt.show()

    try:
        with open(f"results/cyst-model/{model_name}/eval_results.json", "r") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}

    all_results[model_name] = eval["eval_accuracy"]

    os.makedirs(f"results/cyst-model/{model_name}", exist_ok=True)
    with open(f"results/cyst-model/{model_name}/eval_results.json", "w") as f:
        json.dump(all_results, f, indent=4)


