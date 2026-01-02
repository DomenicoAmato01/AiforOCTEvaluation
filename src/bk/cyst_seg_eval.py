import os
import torch   
import pandas as pd
from utils.model import get_segmentation_model
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Compose, NormalizeIntensityd
)
from monai.data import DataLoader, Dataset
from monai.metrics import DiceMetric
import json

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    model, model_name = get_segmentation_model("Cyst")
    model.to(device)

    df = pd.read_excel("data/esaso_eval/cyst_train_test_split.xlsx")
    df_test = df[df['set']=='test']

    images_test_fn = [{
        "image": os.path.join("data/Cyst", f.split('_')[0],'images', f),
        "mask": os.path.join("data/Cyst", f.split('_')[0],'masks', f.split('.')[0] + "m.png")
        } for f in df_test["nomefile"].to_numpy()]
    testset = Dataset(
        data=images_test_fn,
        transform=Compose([
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            NormalizeIntensityd(keys=["image", "mask"], subtrahend=0, divisor=255.0)])
    )

    print(f"Number of testing samples: {len(testset)}")

    test_loader = DataLoader(testset, batch_size=8, shuffle=False)

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", num_classes=1)

    model.load_state_dict(torch.load(f"models/cyst-model/{model_name}.pth"))
    model.eval()

    for test_data in test_loader:
        test_images, test_labels = test_data['image'].to(device), test_data['mask'].to(device)
        with torch.no_grad():
            test_outputs = model.forward(test_images)
            if isinstance(test_outputs, (list, tuple)):
                test_outputs = test_outputs[0]
            test_outputs = torch.sigmoid(test_outputs)
        dice_metric(y_pred=test_outputs, y=test_labels)
    final_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    print(f"Final Dice Score: {final_dice}")

    all_results = {}
    all_results[model_name] = final_dice

    os.makedirs(f"results/cyst-model/{model_name}", exist_ok=True)
    with open(f"results/cyst-model/{model_name}/eval_results.json", "w") as f:
        json.dump(all_results, f, indent=4)