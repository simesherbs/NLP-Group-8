import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer
import torch
import torch.nn as nn
from transformers import BertModel
import numpy as np
from torch.optim import AdamW
import time
df = pd.read_csv('english_movies.csv').head(100)



# Initialize the BERT tokenizer

start_tokenization = time.time()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Convert genres to one-hot encoded labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["genres"])


# Tokenize the document text (use BERT tokenizer)
max_len = 128  # You can adjust this based on your documents

def tokenize_data(text):
    encoding = tokenizer(
        text,
        padding="max_length",  # Padding to ensure uniform length
        truncation=True,  # Truncate if the text exceeds max length
        max_length=max_len,  # Ensure all tokens are the same length
        return_tensors="pt"  # Return as PyTorch tensors
    )
    return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

# Apply tokenization to all documents
input_ids, attention_masks = zip(*df["overview"].astype(str).apply(tokenize_data))

# Convert tokenized text to tensors
input_ids = torch.stack(input_ids)
attention_masks = torch.stack(attention_masks)

end_tokenization = time.time()
print(f"Tokenization runtime: {end_tokenization - start_tokenization:.2f} seconds")

start_model_definition = time.time()
class BERTGenreClassifier(nn.Module):
    def __init__(self, n_labels):
        super(BERTGenreClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        # Forward pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Get the pooled output (representation of the [CLS] token)
        pooled_output = self.dropout(pooled_output)  # Apply dropout
        return self.out(pooled_output)  # Final output layer


from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support

# Create a DataLoader to handle batching
train_dataset = TensorDataset(input_ids, attention_masks, torch.tensor(y, dtype=torch.float32))
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTGenreClassifier(n_labels=y.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

end_model_definition = time.time()
print(f"Model Definition runtime: {end_model_definition - start_model_definition:.2f} seconds")
# Training loop

start_training = time.time()
def train_model(model, dataloader, optimizer, criterion, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

# Train the model
train_model(model, train_dataloader, optimizer, criterion, device)
end_training = time.time()
print(f"Training runtime: {end_training - start_training:.2f} seconds")
start_eval = time.time()
def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs) > 0.5  # Convert logits to binary predictions
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro"
    )
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Evaluate the model
evaluate_model(model, train_dataloader, device)
end_eval=time.time()
print(f"Evaluation runtime: {end_eval - start_eval:.2f} seconds")