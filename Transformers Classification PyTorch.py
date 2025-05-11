import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string
from collections import Counter

# %% Data definition and preprocessing
positive_sentences = [
    "The weather today is absolutely perfect.",
    "I love spending time with my friends.",
    "She always makes me laugh and feel happy.",
    "It was such a rewarding experience to volunteer.",
    "I’ve learned so much from this journey.",
    "His kindness is contagious.",
    "This new project is going to be amazing.",
    "I’m looking forward to what the future holds.",
    "The city looks beautiful in the evening lights.",
    "I’m grateful for the people in my life.",
    "The team’s effort was truly commendable.",
    "We are getting closer to our goals every day.",
    "That was a fun and exciting adventure.",
    "I feel at peace with my decisions.",
    "Her smile brightens up the room.",
    "I’m amazed by the progress I’ve made.",
    "That was such a thoughtful gesture.",
    "I can’t wait for what’s coming next!",
    "It’s so refreshing to be around such positive energy.",
    "I’m so glad to be part of this amazing team.",
    "That was an inspiring conversation.",
    "I love how everything is falling into place.",
    "What a fantastic experience that was!",
    "I’m feeling more confident every day.",
    "I really appreciate your help today."
]

negative_sentences = [
    "I feel like I’m running out of time.",
    "The workload is overwhelming right now.",
    "I’ve been feeling very drained lately.",
    "No one seems to care about my opinions.",
    "Everything feels like an uphill battle right now.",
    "I can’t seem to catch a break.",
    "This project isn’t going anywhere.",
    "It’s been a really tough and challenging day.",
    "I feel like I’m stuck in a rut.",
    "I’m not sure I can keep up with everything.",
    "Things just aren’t going the way I planned.",
    "I’m constantly under a lot of pressure.",
    "This situation feels hopeless.",
    "It’s hard to stay positive with everything going on.",
    "I feel overwhelmed by all the responsibilities.",
    "I’m feeling really disconnected from everyone.",
    "I wish things were different right now.",
    "I don’t know how to fix this.",
    "I feel like I’m not being heard.",
    "I’m exhausted from dealing with constant challenges.",
    "It feels like I’m walking on eggshells all the time.",
    "I’ve been feeling really down lately.",
    "I feel like I’m not making any progress.",
    "There’s so much to do and I’m not sure where to start.",
    "I wish I had more support in this situation."
]

# Preprocess the sentences
def preprocess(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    return text

# Generate dataset
data = positive_sentences + negative_sentences
labels = [1] * len(positive_sentences) + [0] * len(negative_sentences)

# Preprocess the data
data = [preprocess(sentence) for sentence in data]

# Create a vocabulary
def create_vocab(data):
    # Tokenize the sentences
    tokens = [sentence.split() for sentence in data]
    # Flatten the list of tokens
    tokens = [item for sublist in tokens for item in sublist]
    # Create a Counter object to count the frequency of each word
    counter = Counter(tokens)
    # Create a vocabulary dictionary
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(counter.most_common())}
    vocab['<PAD>'] = 0  # Add padding token
    return vocab

max_length = 15

# Transform the data to tensor
def sentence_to_tensor(sentence, vocab, max_length=15):
    # Tokenize the sentence
    tokens = sentence.split()
    # Convert to indices
    indices = [vocab.get(token, 0) for token in tokens]  # 0 for unknown tokens
    # Pad the sequence
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))  # Pad with 0
    else:
        indices = indices[:max_length]  # Truncate to max_length
    return torch.tensor(indices)

# Create the vocabulary
vocab = create_vocab(data)

# Transform sentences to tensors
X = torch.stack([sentence_to_tensor(sentence, vocab) for sentence in data])
y = torch.tensor(labels)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Model definition with dropout
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, num_classes, dropout_rate=0.2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_rate)  # Add Dropout layer
        self.fc = nn.Linear(embedding_dim * max_length, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        output = self.transformer(embedded)
        output = output.view(output.size(0), -1)
        output = torch.relu(self.fc(output))
        output = self.dropout(output)  # Apply Dropout
        output = self.out(output)
        output = self.sigmoid(output)
        return output

# Initialize the model with dropout
model = TransformerClassifier(
    vocab_size=len(vocab), 
    embedding_dim=32, 
    num_heads=4, 
    num_layers=4, 
    hidden_dim=64, 
    num_classes=1,
    dropout_rate=0.2  # Add dropout
)

# Loss and optimizer with weight decay (L2 regularization)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # Add weight decay

# Early stopping parameters
patience = 10
best_loss = float('inf')
patience_counter = 0

# Training loop with early stopping and dropout
number_epochs = 200
for epoch in range(number_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train.long()).squeeze()
    loss = criterion(output, y_train.float())    
    loss.backward()
    optimizer.step()
    
    # Validation step for early stopping
    model.eval()
    with torch.no_grad():
        output_val = model(X_test.long()).squeeze()
        val_loss = criterion(output_val, y_test.float())
        
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0  # Reset the counter
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
    
    print(f"Epoch {epoch+1}/{number_epochs} - Loss: {loss.item():.4f} - Validation Loss: {val_loss.item():.4f}")

# Final evaluation on test set
model.eval()
with torch.no_grad():
    output_test = model(X_test.long()).squeeze()
    predicted_labels = (output_test > 0.5).float()
    accuracy = accuracy_score(y_test.numpy(), predicted_labels.numpy())
    print(f"Test Accuracy: {accuracy:.4f}")
