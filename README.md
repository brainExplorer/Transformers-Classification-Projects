Transformer-based Sentiment Classifier
Introduction
This project is a simple sentiment analysis model built using PyTorch and Transformer architecture. It classifies input sentences as either positive or negative. The dataset consists of 50 handcrafted sentences, evenly split between positive and negative sentiments.
Features
- Transformer-based architecture with custom token embedding
- Dropout regularization
- Early stopping for preventing overfitting
- L2 regularization using weight decay
Dataset
The dataset is manually created and consists of 25 positive and 25 negative sentences. The sentences are preprocessed by removing punctuation and converting to lowercase.
Model Architecture
The model uses a TransformerEncoder with multiple layers and heads. The embedded sentence is passed through the transformer layers, flattened, then passed through a fully connected layer with ReLU activation and dropout, followed by a sigmoid output layer for binary classification.
Training Details
- Optimizer: Adam with learning rate 0.0005 and weight decay 1e-5
- Loss Function: Binary Cross Entropy (BCELoss)
- Max Epochs: 200
- Early Stopping Patience: 10 epochs
- Max Sequence Length: 15
How to Run
1. Install required packages: torch, sklearn
2. Run the Python script to train the model
3. The script automatically splits the data, trains the model with early stopping, and evaluates accuracy on the test set
Output
The final output displays the test accuracy after training.
