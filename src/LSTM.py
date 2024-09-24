
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
import random
import matplotlib.pyplot as plt
from collections import Counter
import re
import seaborn as sns

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# preprocess the text
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


# Calculate the total number of unique words in T
unique_words = set()
for review in X_train_data:
    for word in review.split():
        unique_words.add(preprocess_string(word))
total_unique_words = len(unique_words)

# Calculate the total number of training examples in T
total_training_examples = len(X_train_data)

# Calculate the ratio of positive examples to negative examples in T
positive_examples = sum(y_train_data)
negative_examples = total_training_examples - positive_examples
ratio_positive_to_negative = positive_examples / negative_examples

# Calculate the average length of document in T
document_lengths = [len(review.split()) for review in X_train_data]
average_length_of_document = sum(document_lengths) / total_training_examples

# Calculate the max length of document in T
max_length_of_document = max(document_lengths)

# Print the results
print(f'Total number of unique words in T: {total_unique_words}')
print(f'Total number of training examples in T: {total_training_examples}')
print(f'Ratio of positive examples to negative examples in T: {ratio_positive_to_negative:.2f}')
print(f'Average length of document in T: {average_length_of_document:.2f}')
print(f'Max length of document in T: {max_length_of_document}')


print(f'length of vocabulary {len(vocab)}')

rev_len = [len(i) for i in x_train_]
pd.Series(rev_len).hist()
plt.show()
pd.Series(rev_len).describe()

VOCABULARY_SIZE = 10000
LEARNING_RATE = 1e-3
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


class SentimentRNN(nn.Module):
    def __init__(self, no_layers, output_dim, vocab_size, hidden_dim, embedding_dim, embedding_matrix=None,
                 drop_prob=0.5):
        super(SentimentRNN, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers
        self.vocab_size = vocab_size

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32))
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=no_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sigmoid(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        return sig_out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(DEVICE)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(DEVICE)
        hidden = (h0, c0)
        return hidden

no_layers = 2
vocab_size = len(vocab) + 1
embedding_dim = 100
output_dim = 1
hidden_dim = 256

model = SentimentRNN(no_layers, output_dim, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5)
model.to(DEVICE)

print(model)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def acc(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

clip = 5
valid_loss_min = np.Inf
patience = 4

epoch_tr_loss, epoch_vl_loss = [], []
epoch_tr_acc, epoch_vl_acc = [], []

for epoch in range(EPOCHS):
    train_losses = []
    train_acc = 0.0
    model.train()

    h = model.init_hidden(BATCH_SIZE)
    for batch in train_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        labels = batch['sentiment'].to(DEVICE).long()

        h = tuple([each.data for each in h])

        model.zero_grad()
        output, h = model(input_ids, h)

        loss = criterion(output.squeeze(), labels.float())
        loss.backward()

        train_losses.append(loss.item())

        accuracy = acc(output, labels)
        train_acc += accuracy

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    val_h = model.init_hidden(BATCH_SIZE)
    val_losses = []
    val_acc = 0.0
    model.eval()
    for batch in test_loader:
        val_h = tuple([each.data for each in val_h])

        input_ids = batch['input_ids'].to(DEVICE)
        labels = batch['sentiment'].to(DEVICE).long()

        output, val_h = model(input_ids, val_h)
        val_loss = criterion(output.squeeze(), labels.float())

        val_losses.append(val_loss.item())

        accuracy = acc(output, labels)
        val_acc += accuracy

    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = train_acc / len(train_loader.dataset)
    epoch_val_acc = val_acc / len(test_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(f'Epoch {epoch+1}')
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')

    # Early stopping
    if epoch_val_loss <= valid_loss_min:
        torch.save(model.state_dict(), './state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, epoch_val_loss))
        valid_loss_min = epoch_val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print('Early stopping!')
            break
    print(25 * '==')

fig = plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
plt.plot(epoch_tr_acc, label='Train Acc')
plt.plot(epoch_vl_acc, label='Validation Acc')
plt.title("Accuracy")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(epoch_tr_loss, label='Train loss')
plt.plot(epoch_vl_loss, label='Validation loss')
plt.title("Loss")
plt.legend()
plt.grid()

plt.show()


# Model with pretrained embeddings

embedding_dim = 100
embedding_matrix = load_pretrained_embeddings('../data/all.review.vec.txt', vocab, embedding_dim=embedding_dim)
model_with_emb = SentimentRNN(no_layers, output_dim, vocab_size, hidden_dim, embedding_dim, embedding_matrix, drop_prob=0.5)
model_with_emb.to(DEVICE)

criterion = nn.BCELoss()
optimizer_with_emb = torch.optim.Adam(model_with_emb.parameters(), lr=LEARNING_RATE)

clip = 5
valid_loss_min_with_emb = np.Inf

epoch_tr_loss_with_emb, epoch_vl_loss_with_emb = [], []
epoch_tr_acc_with_emb, epoch_vl_acc_with_emb = [], []

for epoch in range(EPOCHS):
    train_losses_with_emb = []
    train_acc_with_emb = 0.0
    model_with_emb.train()

    h_with_emb = model_with_emb.init_hidden(BATCH_SIZE)
    for batch in train_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        labels = batch['sentiment'].to(DEVICE).long()

        h_with_emb = tuple([each.data for each in h_with_emb])

        model_with_emb.zero_grad()
        output, h_with_emb = model_with_emb(input_ids, h_with_emb)

        loss = criterion(output.squeeze(), labels.float())
        loss.backward()

        train_losses_with_emb.append(loss.item())

        accuracy = acc(output, labels)
        train_acc_with_emb += accuracy

        nn.utils.clip_grad_norm_(model_with_emb.parameters(), clip)
        optimizer_with_emb.step()

    val_h_with_emb = model_with_emb.init_hidden(BATCH_SIZE)
    val_losses_with_emb = []
    val_acc_with_emb = 0.0
    model_with_emb.eval()
    for batch in test_loader:
        val_h_with_emb = tuple([each.data for each in val_h_with_emb])

        input_ids = batch['input_ids'].to(DEVICE)
        labels = batch['sentiment'].to(DEVICE).long()

        output, val_h_with_emb = model_with_emb(input_ids, val_h_with_emb)
        val_loss = criterion(output.squeeze(), labels.float())

        val_losses_with_emb.append(val_loss.item())

        accuracy = acc(output, labels)
        val_acc_with_emb += accuracy

    epoch_train_loss_with_emb = np.mean(train_losses_with_emb)
    epoch_val_loss_with_emb = np.mean(val_losses_with_emb)
    epoch_train_acc_with_emb = train_acc_with_emb / len(train_loader.dataset)
    epoch_val_acc_with_emb = val_acc_with_emb / len(test_loader.dataset)
    epoch_tr_loss_with_emb.append(epoch_train_loss_with_emb)
    epoch_vl_loss_with_emb.append(epoch_val_loss_with_emb)
    epoch_tr_acc_with_emb.append(epoch_train_acc_with_emb)
    epoch_vl_acc_with_emb.append(epoch_val_acc_with_emb)
    print(f'Epoch {epoch+1}')
    print(f'train_loss_with_emb : {epoch_train_loss_with_emb} val_loss_with_emb : {epoch_val_loss_with_emb}')
    print(f'train_accuracy_with_emb : {epoch_train_acc_with_emb*100} val_accuracy_with_emb : {epoch_val_acc_with_emb*100}')

    # Early stopping
    if epoch_val_loss_with_emb <= valid_loss_min_with_emb:
        torch.save(model_with_emb.state_dict(), './state_dict_with_emb.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min_with_emb, epoch_val_loss_with_emb))
        valid_loss_min_with_emb = epoch_val_loss_with_emb
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print('Early stopping!')
            break
    print(25 * '==')



fig = plt.figure(figsize=(20, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(epoch_tr_acc_with_emb, label='Train Acc with Embeddings')
plt.plot(epoch_vl_acc_with_emb, label='Validation Acc with Embeddings')
plt.title("Accuracy with Embeddings")
plt.legend()
plt.grid()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epoch_tr_loss_with_emb, label='Train Loss with Embeddings')
plt.plot(epoch_vl_loss_with_emb, label='Validation Loss with Embeddings')
plt.title("Loss with Embeddings")
plt.legend()
plt.grid()

plt.show()


def predict_text(model, text):
    word_seq = np.array([vocab[preprocess_string(word)] for word in text.split()
                         if preprocess_string(word) in vocab.keys()])
    word_seq = np.expand_dims(word_seq, axis=0)
    pad = torch.from_numpy(padding_(word_seq, 500))
    inputs = pad.to(DEVICE)
    batch_size = 1
    h = model.init_hidden(batch_size)
    h = tuple([each.data for each in h])
    output, h = model(inputs, h)
    return output.item()



