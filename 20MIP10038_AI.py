#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json

# Load the JSON file into a DataFrame
with open('idmanual.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(df.head())


# In[2]:


from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

# Filter out any entries with a status other than 'A'
df = df[df['status'] == 'A']

# Separate descriptions and class IDs
descriptions = df['description'].tolist()
class_ids = df['class_id'].tolist()

# Encode the class labels
label_encoder = LabelEncoder()
encoded_classes = label_encoder.fit_transform(class_ids)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the descriptions
encodings = tokenizer(descriptions, truncation=True, padding=True, return_tensors='pt')


# In[3]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encodings['input_ids'], encoded_classes, test_size=0.2, random_state=42)


# In[4]:


import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, AdamW

class TrademarkDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {'input_ids': self.encodings[idx]}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = TrademarkDataset(X_train, y_train)
test_dataset = TrademarkDataset(X_test, y_test)

# Initialize model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(encoded_classes)))
model.train()

# Training parameters
batch_size = 8
learning_rate = 5e-5
epochs = 3

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        labels = batch['labels']
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'bert_trademark_classifier.pth')


# In[6]:


get_ipython().system('pip install datasets')

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from datasets import load_dataset
from torch.optim import AdamW
from tqdm.auto import tqdm


# In[7]:


get_ipython().system('pip install torch transformers pandas scikit-learn')


# In[16]:


import json

# Load the JSON dataset
file_path = "C:\\Users\\sendm\\Downloads\\idmanual.json"

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load data
data = load_data(file_path)

# Print out the JSON data and its structure
print("First item in the dataset:", data[0])  # Print the first item
print("Type of first item:", type(data[0]))  # Print the type of the first item


# In[17]:


import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the JSON dataset
file_path = "C:\\Users\\sendm\\Downloads\\idmanual.json"

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

# Load data
df = load_data(file_path)

# Print columns and data to understand structure
print("Columns in DataFrame:", df.columns)
print("Sample data:\n", df.head())

# Update with the actual column names if different
description_col = 'text_description'  # Replace with actual column name
classification_col = 'class_code'  # Replace with actual column name

if description_col not in df.columns or classification_col not in df.columns:
    raise ValueError(f"Expected columns '{classification_col}' or '{description_col}' not found in the dataset")

# Encode labels
le = LabelEncoder()
df['encoded_label'] = le.fit_transform(df[classification_col])

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[description_col], df['encoded_label'], test_size=0.2, random_state=42
)

# Initialize tokenizer and parameters
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128  # Adjust based on your needs
batch_size = 16  # Set your batch size

# Create datasets
class TrademarkDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = TrademarkDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len)
val_dataset = TrademarkDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, max_len)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize model
num_labels = len(le.classes_)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Training loop
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3  # Set your number of epochs

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss}')

# Save the model
model.save_pretrained('trademark_model')
tokenizer.save_pretrained('trademark_model')


# In[18]:


import json

# Load the JSON dataset
file_path = "C:\\Users\\sendm\\Downloads\\idmanual.json"

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load data
data = load_data(file_path)

# Print type and a sample to understand the structure
print("Type of data:", type(data))
if isinstance(data, list):
    print("Sample data (first item):", data[0])
elif isinstance(data, dict):
    print("Keys in dictionary:", data.keys())
    print("Sample data (first item):", list(data.values())[0] if data else 'No data')


# In[19]:


import pandas as pd

# Assuming data is a list of dictionaries, convert to DataFrame
if isinstance(data, list):
    df = pd.DataFrame(data)
    print("Columns in DataFrame:", df.columns)
    print("Sample data:\n", df.head())
else:
    print("Data is not a list; unable to convert to DataFrame.")


# In[20]:


# Replace with actual column names after inspection
description_col = 'actual_description_column_name'
classification_col = 'actual_classification_column_name'

# Ensure the columns are present
if description_col not in df.columns or classification_col not in df.columns:
    raise ValueError(f"Expected columns '{classification_col}' or '{description_col}' not found in the dataset")

# Proceed with data encoding and preparation


# In[ ]:


import json
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the JSON dataset
file_path = "C:\\Users\\sendm\\Downloads\\idmanual.json"

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

# Preprocess the data
class TrademarkDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load and preprocess data
df = load_data(file_path)

# Update with the actual column names
description_col = 'description'
classification_col = 'class_id'

# Ensure the columns are present
if description_col not in df.columns or classification_col not in df.columns:
    raise ValueError(f"Expected columns '{classification_col}' or '{description_col}' not found in the dataset")

# Encode labels
le = LabelEncoder()
df['encoded_label'] = le.fit_transform(df[classification_col])

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[description_col], df['encoded_label'], test_size=0.2, random_state=42
)

# Initialize tokenizer and parameters
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128  # Adjust based on your needs
batch_size = 16  # Set your batch size

# Create datasets
train_dataset = TrademarkDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len)
val_dataset = TrademarkDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, max_len)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize model
num_labels = len(le.classes_)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Training loop
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3  # Set your number of epochs

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss}')

# Save the model
model.save_pretrained('trademark_model')
tokenizer.save_pretrained('trademark_model')


# In[ ]:




