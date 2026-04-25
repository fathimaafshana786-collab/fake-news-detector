# ============================================
# PHASE 3 - BERT Fine-tuning (Lighter Version)
# ============================================

# Step 1: Import libraries
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
import os

print("✅ Libraries imported!")

# Step 2: Load cleaned data
print("\nLoading cleaned dataset...")
train_df = pd.read_csv("data/train_clean.csv")
test_df  = pd.read_csv("data/test_clean.csv")

# Use only 5000 samples to make it faster
train_df = train_df.head(5000)
test_df  = test_df.head(1000)

print(f"Train samples : {len(train_df)}")
print(f"Test samples  : {len(test_df)}")

# Step 3: Load BERT Tokenizer
print("\nLoading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("✅ Tokenizer loaded!")

# Step 4: Create Dataset class
class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df         = df
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text  = str(self.df.iloc[idx]['content'])
        label = int(self.df.iloc[idx]['label'])

        encoding = self.tokenizer(
            text,
            max_length     = self.max_length,
            padding        = 'max_length',
            truncation     = True,
            return_tensors = 'pt'
        )

        return {
            'input_ids'      : encoding['input_ids'].squeeze(),
            'attention_mask' : encoding['attention_mask'].squeeze(),
            'label'          : torch.tensor(label, dtype=torch.long)
        }

# Step 5: Create DataLoaders
print("\nPreparing data loaders...")
train_dataset = NewsDataset(train_df, tokenizer)
test_dataset  = NewsDataset(test_df,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)

print("✅ Data loaders ready!")

# Step 6: Load BERT Model
print("\nLoading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2
)

# Step 7: Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
print("✅ BERT model loaded!")

# Step 8: Setup optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Step 9: Training function
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds  = []
    all_labels = []

    for batch_idx, batch in enumerate(loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}")

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy

# Step 10: Evaluation function
def evaluate(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds)
    return accuracy, f1

# Step 11: Train the model
print("\n🚀 Starting BERT Training...")
print("=" * 50)

EPOCHS = 1

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 30)

    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, device
    )

    print(f"\nTrain Loss     : {train_loss:.4f}")
    print(f"Train Accuracy : {train_acc:.4f}")

    val_acc, val_f1 = evaluate(model, test_loader, device)
    print(f"Test Accuracy  : {val_acc:.4f}")
    print(f"Test F1 Score  : {val_f1:.4f}")

print("\n" + "=" * 50)
print("✅ Training Complete!")

# Step 12: Save the model
print("\nSaving model...")
os.makedirs("bert_model", exist_ok=True)
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")
print("✅ Model saved to bert_model folder!")
print("\n🎉 Phase 3 Complete!")