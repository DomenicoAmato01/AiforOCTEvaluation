import torch
from utils.model import get_segmentation_model
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
import os
import pandas as pd
# from torchvision.transforms import Resize

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Compose, NormalizeIntensityd
)
from monai.data import DataLoader, Dataset

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    model, model_name = get_segmentation_model("Cyst")

    df = pd.read_excel("data/esaso_eval/cyst_train_test_split.xlsx")
    df_train = df[df['set']=='train']
    df_test = df[df['set']=='test']
    fn = df_train["nomefile"].to_numpy()

    images_train_fn = [{
        "image": os.path.join("data/Cyst", f.split('_')[0],'images', f),
        "mask": os.path.join("data/Cyst", f.split('_')[0],'masks', f.split('.')[0] + "m.png")
        } for f in fn]
    # masks_train_fn = [os.path.join("data/Cyst", f.split('_')[0],'masks', f.split('.')[0] + "m.png") for f in fn]
    # images_train = dict(image=images_train_fn, mask=masks_train_fn)
    trainset = Dataset(
        data=images_train_fn,
        transform=Compose([
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            NormalizeIntensityd(keys=["image", "mask"], subtrahend=0, divisor=255.0)])
    )
    # trainset = ImageDataset(
    #     image_files=images_train_fn,
    #     seg_files=masks_train_fn,
    #     transform=Resize((1, 512,512)),
    #     seg_transform=Resize((1, 512,512))
    # )

    print(f"Number of training samples: {len(trainset)}")

    images_test_fn = [{
        "image": os.path.join("data/Cyst", f.split('_')[0],'images', f),
        "mask": os.path.join("data/Cyst", f.split('_')[0],'masks', f.split('.')[0] + "m.png")
        } for f in df_test["nomefile"].to_numpy()]
    # images_test_fn = [os.path.join("data/Cyst", f.split('_')[0],'images', f) for f in df_test["nomefile"].to_numpy()]
    # masks_test_fn = [os.path.join("data/Cyst", f.split('_')[0],'masks', f) for f in df_test["nomefile"].to_numpy()]
    # images_test = dict(image=images_test_fn, mask=masks_test_fn)
    testset = Dataset(
        data=images_test_fn,
        transform=Compose([
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            NormalizeIntensityd(keys=["image", "mask"], subtrahend=0, divisor=255.0)])
    )

    print(f"Number of testing samples: {len(testset)}")

    model.to(device)

    train_loader = DataLoader(trainset, batch_size=8, shuffle=True)
    test_loader = DataLoader(testset, batch_size=8, shuffle=False)

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", num_classes=1)

    loss_function = DiceLoss(sigmoid=False)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    best_loss = float('inf')
    no_improve_epochs = 0
    epochs = 100
    patience = 10
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for i, data in enumerate(train_loader, 0):
            step = i + 1
            # print("Loading batch data")
            inputs, labels = data['image'].to(device), data['mask'].to(device)
            # print(f"Input batch shape: {inputs.shape}, Label batch shape: {labels.shape}")
            optimizer.zero_grad()
            # print("Performing forward pass")
            outputs = model(inputs)
            if(isinstance(outputs, list)):
                outputs = outputs[0]
            outputs = torch.sigmoid(outputs)
            loss = loss_function(outputs, labels)
            # print(f"Computed loss: {loss.item()}")
            # print("Performing backward pass")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()   
            epoch_len = len(trainset) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}", end='\r')
        epoch_loss /= step
        
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            # reset the status for next validation round
            dice_metric.reset()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in test_loader:
                    val_images, val_labels = val_data['image'].to(device), val_data['mask'].to(device)
                    val_outputs = model.forward(val_images)
                    if(isinstance(val_outputs, list)):
                        val_outputs = val_outputs[0]
                    val_outputs = torch.sigmoid(val_outputs)
                    # val_outputs = (val_outputs > 0.5).float()

                    dice_metric(y_pred=val_outputs, y=val_labels)
                    
                # aggregate the final mean dice result
                loss_val = loss_function(val_outputs, val_labels).item()
                print(f"validation loss: {loss_val:.4f}")
                if loss_val < best_loss:
                    best_loss = loss_val
                    no_improve_epochs = 0
                    print("New best validation loss, resetting no improvement counter")
                else:
                    no_improve_epochs += 1 
                    torch.save(model.state_dict(), os.path.join("models", "cyst-model", f"{model_name}.pth"))
                    print("saved new best model")
                    print(f"No improvement epochs: {no_improve_epochs}/{patience}")
                if no_improve_epochs >= patience:
                    print("Early stopping triggered")
                    break
                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )

                
                

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
