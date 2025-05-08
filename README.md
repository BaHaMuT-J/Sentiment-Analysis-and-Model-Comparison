# Sentiment Analysis and Model Comparison

## Overview

This project is a personal learning exercise to explore foundational concepts in Natural Language Processing (NLP), including text preprocessing, vocabulary construction, vectorization, and the implementation of various neural network architectures. Using the [IMDB movie reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz), the goal is to perform sentiment analysis and compare the performance of multiple models implemented in PyTorch.

## Dataset

The IMDB dataset (Maas et al., 2011) contains 50,000 movie reviews labeled as positive or negative. For training purposes, I sampled half of the positive and negative reviews from the test set to augment the training set, resulting in a total training size of 75% of the full dataset.

## Data Preprocessing

- **Text Cleaning & Tokenization**: All reviews are lowercased, stripped of non-alphabetic characters and stopwords, and tokenized using [NLTK](https://www.nltk.org/).
- **Vocabulary**: A vocabulary of the top 8,000 most frequent words in the training set is created.
- **Vectorization**:
  - One-hot encoding is used for the Feedforward Neural Network (FNN).
  - Indexed sequences are used for all RNN-based models.
- **Data Loading**: A custom PyTorch `Dataset` class is created, and `DataLoader` is used with a specified `BATCH_SIZE`.

## Model Architectures

Four different models were implemented and trained for comparison:

### 1. Feedforward Neural Network (FNN)

- Input: One-hot encoded vectors
- Architecture: No hidden layer
- Optimizer: Adam
- Learning Rate: 0.0001

### 2. Recurrent Neural Network (RNN)

- Input: Embedded word indices (Embedding size: 100)
- Architecture: One bidirectional RNN layer with 128 hidden units
- Regularization: Dropout (`p=0.5`)
- Optimizer: Adam
- Learning Rate: 0.001

### 3. Gated Recurrent Unit (GRU)

- Input: Embedded word indices (Embedding size: 100)
- Architecture: One bidirectional GRU layer with 128 hidden units
- Regularization: Dropout (`p=0.5`)
- Optimizer: Adam
- Learning Rate: 0.001

### 4. Long Short-Term Memory (LSTM)

- Input: Embedded word indices (Embedding size: 100)
- Architecture: One bidirectional LSTM layer with 128 hidden units
- Regularization: Dropout (`p=0.5`)
- Optimizer: Adam
- Learning Rate: 0.001

### Training Configuration

- Epochs: 10
- Batch Size: 32

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
| ----- | -------- | --------- | ------ | -------- |
| FNN   | 0.8913   | 0.8869    | 0.8967 | 0.8918   |
| RNN   | 0.5135   | 0.5128    | 0.5505 | 0.5309   |
| GRU   | 0.8580   | 0.8488    | 0.8711 | 0.8598   |
| LSTM  | 0.8260   | 0.8432    | 0.8012 | 0.8217   |

## Conclusion

The results highlight that when the task does not require capturing the sequential nature of text—such as when using a bag-of-words or one-hot representation—Feedforward Neural Networks (FNNs) can perform remarkably well. In this case, the FNN outperformed more complex recurrent models like RNNs, GRUs, and LSTMs. This suggests that for certain sentiment classification tasks, especially when global word presence is more informative than word order, simpler architectures may be both effective and computationally efficient. Meanwhile, models like GRU and LSTM still demonstrated strong performance, showing their strength in tasks where sequence and context are more critical. The basic RNN, however, underperformed due to its limitations in capturing long-range dependencies and vanishing gradient issues.

## Citation

> Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In _Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies_ (pp. 142–150). Association for Computational Linguistics. http://www.aclweb.org/anthology/P11-1015
