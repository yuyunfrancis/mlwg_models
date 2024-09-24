
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import os
import random
import matplotlib.pyplot as plt
from collections import Counter
import re
import seaborn as sns

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Using {DEVICE}')

def get_data(path, sentiment):
    x = []  # list of reviews
    y = []  # labels, where 1 represents a positive review and 0 negative

    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r') as f:
            x.append(f.read())
            y.append(sentiment)

    return x, y

train_data_path_P = '../data/train/positive'
train_data_path_N = '../data/train/negative'

X_train_P, y_train_P = get_data(train_data_path_P, 1)
X_train_N, y_train_N = get_data(train_data_path_N, 0)

X_train = X_train_P + X_train_N
y_train = y_train_P + y_train_N

# Combine data into a list of tuples
combined = list(zip(X_train, y_train))

# Shuffle the combined list
random.shuffle(combined)

# Unpack the shuffled data
X_train, y_train = zip(*combined)

# Train dataframe
train_df = pd.DataFrame({'review': X_train, 'sentiment': y_train})
train_df.head()

X_train_data, y_train_data = train_df['review'].values, train_df['sentiment'].values
print(f'shape of train data is {X_train_data.shape}')

# Test data
test_data_path_P = '../data/test/positive'
test_data_path_N = '../data/test/negative'

X_test_P, y_test_P = get_data(test_data_path_P, 1)
X_test_N, y_test_N = get_data(test_data_path_N, 0)

X_test = X_test_P + X_test_N
y_test = y_test_P + y_test_N

# Combine data into a list of tuples
combined = list(zip(X_test, y_test))

# Shuffle the combined list
random.shuffle(combined)

# Unpack the shuffled data
X_test, y_test = zip(*combined)

# Test dataframe
test_df = pd.DataFrame({'review': X_test, 'sentiment': y_test})
test_df.tail()

X_test_data, y_test_data = test_df['review'].values, test_df['sentiment'].values
print(f'shape of test data is {X_test_data.shape}')

dd = pd.Series(y_train_data).value_counts()
sns.barplot(x=np.array(['0', '1']), y=dd.values)
plt.show()


custom_stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}


def preprocess_string(s):
    s = re.sub(r"[^\w\s]", '', s)
    s = re.sub(r"\s+", ' ', s)
    s = re.sub(r"\d", '', s)
    return s

def tokenize(x_train, y_train, x_val, y_val):
    word_list = []
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in custom_stopwords and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:10000]
    onehot_dict = {w: i+1 for i, w in enumerate(corpus_)}

    final_list_train, final_list_test = [], []
    for sent in x_train:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])

    encoded_train = [1 if label == 1 else 0 for label in y_train]
    encoded_test = [1 if label == 1 else 0 for label in y_val]
    return np.array(final_list_train, dtype=object), np.array(encoded_train), np.array(final_list_test, dtype=object), np.array(encoded_test), onehot_dict


#  Tokenize the data  and get the vocabulary
x_train_, y_train_, x_test_, y_test_, vocab = tokenize(X_train_data, y_train_data, X_test_data, y_test_data)


def load_pretrained_embeddings(filepath, vocab, embedding_dim=100):
    embeddings_index = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(vocab)+1, embedding_dim))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

    return embedding_matrix


rev_len = [len(i) for i in x_train_]
pd.Series(rev_len).hist()
plt.show()
pd.Series(rev_len).describe()

VOCABULARY_SIZE = 10000
LEARNING_RATE = 1e-4
BATCH_SIZE = 100
EPOCHS = 10

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

x_train_pad = padding_(x_train_, 500)
x_test_pad = padding_(x_test_, 500)


class AmazonReviewsDataset(Dataset):
    def __init__(self, reviews, sentiments):
        self.reviews = reviews
        self.sentiments = sentiments

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review = self.reviews[index]
        sentiment = self.sentiments[index]
        return {
            'input_ids': torch.tensor(review, dtype=torch.long),
            'sentiment': torch.tensor(sentiment, dtype=torch.long)
        }

train_dataset = AmazonReviewsDataset(x_train_pad, y_train_)
test_dataset = AmazonReviewsDataset(x_test_pad, y_test_)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

dataiter = iter(train_loader)
sample_batch = next(dataiter)

sample_x = sample_batch['input_ids']
sample_y = sample_batch['sentiment']

print('Sample input size: ', sample_x.size())
print('Sample input: \n', sample_x)
print('Sample sentiment: \n', sample_y)


class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, embedding_matrix=None):
        super().__init__()

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32),
                                                          freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return torch.sigmoid(self.fc(cat))


VOCABULARY_SIZE = len(vocab) + 1
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3

# Initialize the model
model = SentimentCNN(VOCABULARY_SIZE, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
model.to(DEVICE)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def acc(pred, label):
    rounded_pred = torch.round(pred)
    correct = (rounded_pred == label).float()
    return correct.sum() / len(correct)


# Training loop
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch['input_ids'].to(DEVICE)).squeeze(1)
        loss = criterion(predictions, batch['sentiment'].float().to(DEVICE))
        acc_ = acc(predictions, batch['sentiment'].to(DEVICE))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc_.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# Evaluation loop
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch['input_ids'].to(DEVICE)).squeeze(1)
            loss = criterion(predictions, batch['sentiment'].float().to(DEVICE))
            acc_ = acc(predictions, batch['sentiment'].to(DEVICE))
            epoch_loss += loss.item()
            epoch_acc += acc_.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# Training
train_losses, train_accs = [], []
val_losses, val_accs = [], []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, test_loader, criterion)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'cnn-model.pt')

    print(f'Epoch: {epoch + 1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')

    print(25 * '==')

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Losses')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title('Accuracies')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


EPOCHS = 10
# Model with pretrained embeddings
embedding_matrix = load_pretrained_embeddings('../data/all.review.vec.txt', vocab, EMBEDDING_DIM)
model_with_emb = SentimentCNN(VOCABULARY_SIZE, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, embedding_matrix)
model_with_emb.to(DEVICE)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model_with_emb.parameters(), lr=LEARNING_RATE)

# Training
train_with_emb_losses, train_with_emb_accs = [], []
val_with_emb_losses, val_with_emb_accs = [], []

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model_with_emb, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model_with_emb, test_loader, criterion)

    train_with_emb_losses.append(train_loss)
    train_with_emb_accs.append(train_acc)
    val_with_emb_losses.append(val_loss)
    val_with_emb_accs.append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model_with_emb.state_dict(), 'cnn-model-with-emb.pt')

    print(f'Epoch with Embeddings: {epoch + 1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')

    print(25 * '==')

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_with_emb_losses, label='Train Loss with Embeddings')
plt.plot(val_with_emb_losses, label='Validation Loss with Embeddings')
plt.title('Losses')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_with_emb_accs, label='Train Accuracy with Embeddings')
plt.plot(val_with_emb_accs, label='Validation Accuracy with Embeddings')
plt.title('Accuracies')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()



# Function to predict sentiment for a given text
def predict_sentiment(model, text):
    model.eval()
    tokenized = [vocab[preprocess_string(word)] for word in text.split() if preprocess_string(word) in vocab]
    if len(tokenized) < 500:
        tokenized = tokenized + [0] * (500 - len(tokenized))
    else:
        tokenized = tokenized[:500]
    tensor = torch.LongTensor(tokenized).unsqueeze(0).to(DEVICE)
    prediction = model(tensor).squeeze(1)
    return prediction.item()



