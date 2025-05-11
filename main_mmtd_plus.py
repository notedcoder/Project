
import os
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from dataset_mmtd_plus import EmailDatasetMMTDPlus
from models_mmtd_plus import MMTDPlus
from losses import FocalLoss

# Hardcoded file paths for your system
CSV_PATH = "/Users/abhinandan/Desktop/PEOJECT/MMTD++_Enhanced_Code/edp_dataset.csv"
IMG_DIR = "/Users/abhinandan/Desktop/PEOJECT/MMTD++_Enhanced_Code/pics"
TEXT_MODEL = "bert-base-multilingual-cased"
IMAGE_MODEL = "google/vit-base-patch16-224"
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5

def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for ids, mask, img, lang_id, label in loader:
        ids, mask, img, lang_id, label = ids.to(device), mask.to(device), img.to(device), lang_id.to(device), label.to(device)
        optimizer.zero_grad()
        logits = model(ids, mask, img, lang_id)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for ids, mask, img, lang_id, label in loader:
            ids, mask, img, lang_id = ids.to(device), mask.to(device), img.to(device), lang_id.to(device)
            logits = model(ids, mask, img, lang_id)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(label.tolist())
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['ham', 'spam']))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = EmailDatasetMMTDPlus(CSV_PATH, IMG_DIR, TEXT_MODEL, IMAGE_MODEL)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = MMTDPlus(text_model=TEXT_MODEL, image_model=IMAGE_MODEL).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = FocalLoss(alpha=1.0, gamma=2.0)

    print("✅ Starting training...")
    for epoch in range(EPOCHS):
        loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {loss:.4f}")

    print("✅ Evaluation:")
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()