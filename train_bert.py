# ============================================
# IMPROVED BERT Training - Balanced Version
# ============================================

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

print("✅ Libraries imported!")

# Load cleaned data
print("\nLoading cleaned dataset...")
train_df = pd.read_csv("data/train_clean.csv")
test_df  = pd.read_csv("data/test_clean.csv")

# ============================================
# IMPROVEMENT 1 — Balance the dataset
# ============================================
fake_train = train_df[train_df['label'] == 1]
real_train = train_df[train_df['label'] == 0]

print(f"\nBefore balancing:")
print(f"Fake samples : {len(fake_train)}")
print(f"Real samples : {len(real_train)}")

# Take equal samples from both
min_samples = min(len(fake_train), len(real_train))
fake_train  = fake_train.sample(min_samples, random_state=42)
real_train  = real_train.sample(min_samples, random_state=42)

# Combine and shuffle
train_df = pd.concat([fake_train, real_train])
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nAfter balancing:")
print(f"Fake samples : {len(fake_train)}")
print(f"Real samples : {len(real_train)}")
print(f"Total        : {len(train_df)}")

# ============================================
# IMPROVEMENT 2 — Use more data
# ============================================
# Use 10000 samples instead of 5000
train_df = train_df.head(10000)
test_df  = test_df.head(2000)

print(f"\nFinal dataset:")
print(f"Train : {len(train_df)}")
print(f"Test  : {len(test_df)}")

# Load tokenizer
print("\nLoading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("✅ Tokenizer loaded!")

# Dataset class
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

# DataLoaders
print("\nPreparing data loaders...")
train_dataset = NewsDataset(train_df, tokenizer)
test_dataset  = NewsDataset(test_df,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)
print("✅ Data loaders ready!")

# Load BERT
print("\nLoading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
print("✅ BERT model loaded!")

# ============================================
# IMPROVEMENT 3 — Better optimizer settings
# ============================================
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Training function
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

        # ============================================
        # IMPROVEMENT 4 — Gradient clipping
        # ============================================
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}")

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy

# Evaluation function
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
    f1       = f1_score(all_labels, all_preds, average='weighted')

    # ============================================
    # IMPROVEMENT 5 — Detailed report
    # ============================================
    report = classification_report(
        all_labels, all_preds,
        target_names=['Real News', 'Fake News']
    )
    return accuracy, f1, report

# ============================================
# IMPROVEMENT 6 — Train for 2 epochs
# ============================================
print("\n🚀 Starting Improved BERT Training...")
print("=" * 50)

EPOCHS = 2
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 30)

    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, device
    )

    print(f"\nTrain Loss     : {train_loss:.4f}")
    print(f"Train Accuracy : {train_acc:.4f}")

    val_acc, val_f1, report = evaluate(model, test_loader, device)
    print(f"Test Accuracy  : {val_acc:.4f}")
    print(f"Test F1 Score  : {val_f1:.4f}")
    print(f"\nDetailed Report:\n{report}")

    # Save best model
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        print("💾 New best model! Saving...")
        os.makedirs("bert_model", exist_ok=True)
        model.save_pretrained("bert_model")
        tokenizer.save_pretrained("bert_model")
        print("✅ Best model saved!")

print("\n" + "=" * 50)
print(f"🏆 Best Accuracy : {best_accuracy:.4f}")
print("✅ Improved Training Complete!")
print("🎉 Your model is now more balanced!")