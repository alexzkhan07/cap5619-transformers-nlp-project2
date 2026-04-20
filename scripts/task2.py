import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
DATA_FILE = "./data/raw/amazon_reviews.csv"
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128      # Maximum length of the review text to process
BATCH_SIZE = 16    # Number of reviews processed at a time
EPOCHS = 2         # Kept low (2) strictly to avoid over-finetuning!

# --- 1. Custom PyTorch Dataset ---
class AmazonReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        label = self.labels[item]

        # Tokenize the text, add [CLS] and [SEP], and pad/truncate
        # Using the standard __call__ method instead of encode_plus
        encoding = self.tokenizer(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

# --- 2. Main Training and Evaluation Pipeline ---
def main():
    print("Loading data...")
    # Load data
    df = pd.read_csv(DATA_FILE)
    
    # We only need the text and the rating
    df = df[['reviewText', 'overall']].dropna()
    
    # PyTorch expects labels to be 0-indexed integers. 
    # Amazon ratings are 1.0 to 5.0, so we subtract 1 to get 0 to 4.
    df['label'] = df['overall'].astype(int) - 1
    
    # Split the dataset: 80% Training, 20% Testing
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training set size: {len(df_train)}")
    print(f"Testing set size: {len(df_test)}")

    print("Loading Tokenizer and Model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Load BERT with a classification head specifically for 5 classes
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Create DataLoaders
    train_dataset = AmazonReviewDataset(df_train['reviewText'].to_numpy(), df_train['label'].to_numpy(), tokenizer, MAX_LEN)
    test_dataset = AmazonReviewDataset(df_test['reviewText'].to_numpy(), df_test['label'].to_numpy(), tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Setup Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # --- 3. Training Loop ---
    for epoch in range(EPOCHS):
        print(f"\n======== Epoch {epoch + 1} / {EPOCHS} ========")
        model.train()
        total_train_loss = 0
        
        # tqdm adds a nice progress bar to the terminal
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            # Move batch data to the device (GPU/CPU)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            # Clear previous gradients
            model.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )
            
            # The model automatically calculates CrossEntropyLoss when labels are provided
            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward pass & Optimizer step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevent exploding gradients
            optimizer.step()
            
            # Update progress bar
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

    # --- 4. Evaluation Loop ---
    print("\n======== Running Evaluation on Test Set ========")
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Get the predicted class (0 to 4) by finding the highest logit score
            _, preds = torch.max(outputs.logits, dim=1)
            
            correct_predictions += torch.sum(preds == targets).item()
            total_predictions += targets.size(0)

    accuracy = correct_predictions / total_predictions
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Correctly predicted {correct_predictions} out of {total_predictions} reviews.")

if __name__ == "__main__":
    main()