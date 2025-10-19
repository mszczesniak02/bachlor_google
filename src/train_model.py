import torch

from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp #preset models


from hyper import *         # hyperparamethers for the model
from datasets import *      # custom datasets methods and funcs

import numpy as np


class DiceLoss(torch.nn.Module):
  def __init__(self, smooth=1e-6):
    super(DiceLoss,self).__init__()
    self.smooth = smooth
  def forward(self, predictions, targets):
    predictions = torch.sigmoid(predictions)

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()
    dice = (2. * intersection  + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

    return 1-dice

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = .0

    for batch_idx, (images, masks) in enumerate(train_loader):
        # Przenieś na GPU
        # images = images.to(device)
        # masks = masks.to(device)
        
        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader, criterion,device):
  model.eval()
  running_loss = 0.0

  with torch.no_grad():
    for images,masks in val_loader:
      images = images.to(device)
      masks = masks.to(device)

      predictions = model(images)
      loss = criterion(predictions, masks)

      running_loss += loss.item()
  
  avg_loss = running_loss / len(val_loader)
  return avg_loss

def epoch_stat(epoch_times:list):
    m = epoch_stat[::-1]
    for i, t in enumerate(m):
        if (i +1 == len(epoch_times)):
            break
        m[i] -= m[i+1]
    return m[::-1]  


def make_checkpoint(model, optimizer, best_val_loss, training_loss):
   torch.save({
      'epoch': 3,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'train_loss': 0.8946,
      'val_loss': best_val_loss,} , '../models/unet_epoch.pth')



def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Kompletna ewaluacja modelu
    
    Returns:
        metrics: dict z metrykami
        predictions: array predykcji [N, H, W]
        ground_truths: array prawdziwych masek [N, H, W]
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    print("🔍 Running evaluation...")
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Processing batches"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            
            # Do CPU
            all_predictions.append(predictions.cpu().numpy())
            all_ground_truths.append(masks.cpu().numpy())
    
    # Concatenate
    predictions = np.concatenate(all_predictions, axis=0)[:, 0]  # [N, H, W]
    ground_truths = np.concatenate(all_ground_truths, axis=0)[:, 0]  # [N, H, W]
    
    # Binaryzacja
    pred_binary = (predictions > threshold).astype(np.float32)
    gt_binary = (ground_truths > 0.5).astype(np.float32)
    
    # ========================================
    # OBLICZ METRYKI
    # ========================================
    ious = []
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    for pred, gt in zip(pred_binary, gt_binary):
        # Confusion matrix
        tp = (pred * gt).sum()
        fp = (pred * (1 - gt)).sum()
        fn = ((1 - pred) * gt).sum()
        tn = ((1 - pred) * (1 - gt)).sum()
        
        # IoU (Intersection over Union)
        iou = tp / (tp + fp + fn + 1e-6)
        ious.append(iou)
        
        # Precision (jakość detekcji)
        precision = tp / (tp + fp + 1e-6)
        precisions.append(precision)
        
        # Recall (czułość)
        recall = tp / (tp + fn + 1e-6)
        recalls.append(recall)
        
        # F1 Score (harmonic mean)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        f1_scores.append(f1)
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        accuracies.append(accuracy)
    
    metrics = {
        'iou_mean': np.mean(ious),
        'iou_std': np.std(ious),
        'iou_median': np.median(ious),
        'precision_mean': np.mean(precisions),
        'precision_std': np.std(precisions),
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'accuracy_mean': np.mean(accuracies),
        'ious': ious,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
    }
    
    return metrics, predictions, ground_truths

def main()-> int:

    dataset = get_dataset(IMAGE_PATH, MASK_PATH)
    [train_set, test_set, val_set] = split_dataset(dataset, .8, .1, .1)

    train_loader = DataLoader( train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader( test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader( val_set, batch_size=BATCH_SIZE , shuffle=False, num_workers=WORKERS    , pin_memory=PIN_MEMORY)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )

    model = model.to(device)       


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(60*'.')
    print(f"\n Model size:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1e6:.1f} MB")



    criterion = DiceLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    print(60*'.')
    print("\n Optimizer:")
    print(f"   Type: Adam")
    print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"   Weight decay: {optimizer.param_groups[0]['weight_decay']}")
    print(60*'.')




    epochs = EPOCHS
    best_val_loss = float('inf')

    epochs_time = []

    loop = tqdm(range(epochs), unit="Epoch", leave=True)

    for epoch in loop:
        

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        epochs_time.append(loop.format_dict['elapsed'])

        loop.set_description(f"Current Loss:{best_val_loss}" )
        loop.update()

    loop.close()
    
    epochs_total_time = epochs_time[-1]
    epochs_time = epoch_stat(epochs_time)        

    print("TRAINING COMPLETED!")
    print(f"Best Val Loss: {best_val_loss:.4f}")

    print("\t Training time per epoch:")
    [print(f"epoch {i+1} {epochs_time[i] }s") for i in range(len(epochs_time))]

    make_checkpoint(model, optimizer, best_val_loss, train_loss)

    metrics, predictions, ground_truths = evaluate_model(model, test_loader, device)

    print(metrics)

    return 0



if __name__ == "__main__":
  main()
else:
  pass

