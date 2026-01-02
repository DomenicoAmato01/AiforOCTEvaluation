import torch
from utils.model import get_segmentation_model
from monai.data import ImageDataset, DataLoader
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
import os
import pandas as pd
# from torchvision.transforms import Resize

from monai.transforms import Compose, Resize
    
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    model, model_name = get_segmentation_model("Cyst")

    df = pd.read_excel("data/esaso_eval/cyst_train_test_split.xlsx")
    df_train = df[df['set']=='train']
    df_test = df[df['set']=='test']
    fn = df_train["nomefile"].to_numpy()

    images_train_fn = [os.path.join("data/Cyst", f.split('_')[0],'images', f) for f in fn]
    masks_train_fn = [os.path.join("data/Cyst", f.split('_')[0],'masks', f.split('.')[0] + "m.png") for f in fn]

    trainset = ImageDataset(
        image_files=images_train_fn,
        seg_files=masks_train_fn,
        transform=Resize((1, 512,512)),
        seg_transform=Resize((1, 512,512))
    )

    print(f"Number of training samples: {len(trainset)}")

    images_test_fn = [os.path.join("data/Cyst", f.split('_')[0],'images', f) for f in df_test["nomefile"].to_numpy()]
    masks_test_fn = [os.path.join("data/Cyst", f.split('_')[0],'masks', f) for f in df_test["nomefile"].to_numpy()]
    testset = ImageDataset(
        image_files=images_test_fn,
        seg_files=masks_test_fn,
        transform=Resize((1, 512,512)),
        seg_transform=Resize((1, 512,512))
    )

    print(f"Number of testing samples: {len(testset)}")

    model.to(device)

    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    for epoch in range(10):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{10}")
        model.train()
        epoch_loss = 0
        step = 0
        for i, data in enumerate(train_loader, 0):
            step = i + 1
            print("Loading batch data")
            inputs, labels = data[0].to(device), data[1].to(device)
            print(f"Input batch shape: {inputs.shape}, Label batch shape: {labels.shape}")
            optimizer.zero_grad()
            print("Performing forward pass")
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            print(f"Computed loss: {loss.item()}")
            print("Performing backward pass")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(trainset) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in test_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model.forward(val_images)
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_segmentation2d_array.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
