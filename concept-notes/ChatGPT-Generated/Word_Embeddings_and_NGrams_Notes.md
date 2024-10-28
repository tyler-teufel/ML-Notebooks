
# Word Embeddings and N-Grams: A Beginner's Guide

## Table of Contents
1. [Introduction](#introduction)
2. [N-Grams](#n-grams)
   - [What are N-Grams?](#what-are-n-grams)
   - [Examples of N-Grams](#examples-of-n-grams)
   - [Generating N-Grams in Python](#generating-n-grams-in-python)
   - [Applications of N-Grams](#applications-of-n-grams)
3. [Word Embeddings](#word-embeddings)
   - [What are Word Embeddings?](#what-are-word-embeddings)
   - [Word2Vec](#word2vec)
   - [FastText](#fasttext)
   - [Using Pre-trained Embeddings from Kaggle](#using-pre-trained-embeddings-from-kaggle)
   - [Applications of Word Embeddings](#applications-of-word-embeddings)
4. [Conclusion](#conclusion)

## Introduction

Understanding how machines process language is crucial for building modern Natural Language Processing (NLP) models. Two foundational concepts in this area are **N-Grams** and **Word Embeddings**. N-Grams focus on understanding sequences of words, while word embeddings represent words in continuous vector spaces that encode semantic meaning.

---

## N-Grams

### What are N-Grams?

N-Grams are contiguous sequences of `N` items from a given sample of text or speech. These items can be words, characters, or even syllables. 

- **Unigram**: A single word (N=1)
- **Bigram**: A sequence of two words (N=2)
- **Trigram**: A sequence of three words (N=3)
- **N-Gram**: A sequence of N words

### Examples of N-Grams

Let's consider the sentence:
> "Natural language processing is fun."

- **Unigrams**: ['Natural', 'language', 'processing', 'is', 'fun']
- **Bigrams**: ['Natural language', 'language processing', 'processing is', 'is fun']
- **Trigrams**: ['Natural language processing', 'language processing is', 'processing is fun']

### Generating N-Grams in Python

You can generate N-Grams using Python's `nltk` library or simply by using list comprehensions.

```python
from nltk import ngrams
from collections import Counter

sentence = "Natural language processing is fun."
n = 2  # For bigrams
bigrams = ngrams(sentence.split(), n)
print(list(bigrams))

# Output: [('Natural', 'language'), ('language', 'processing'), ('processing', 'is'), ('is', 'fun')]
```

For creating a custom function to generate N-Grams without `nltk`:

```python
def generate_ngrams(text, n):
    words = text.split()
    return [words[i:i+n] for i in range(len(words)-n+1)]

text = "Natural language processing is fun."
print(generate_ngrams(text, 3))  # For trigrams
```

### Applications of N-Grams

1. **Text Prediction**: N-Grams are used in predictive text applications such as mobile keyboards to suggest the next word based on the preceding N-1 words.
2. **Machine Translation**: N-Grams help improve the fluency and coherence of translated text by preserving common sequences of words.
3. **Plagiarism Detection**: N-Gram-based models can detect plagiarized content by comparing sequences across different documents.

---

## Word Embeddings

### What are Word Embeddings?

Word embeddings represent words as dense vectors of fixed size, where each word is mapped to a continuous vector space. These embeddings encode semantic similarity, i.e., words with similar meanings are represented by similar vectors. 

Popular embedding techniques include:
- **Word2Vec**
- **FastText**
- **GloVe**

### Word2Vec

**Word2Vec** is a neural network model that learns word embeddings based on the context of words. It comes in two main variants:
1. **CBOW (Continuous Bag of Words)**: Predicts the current word based on its surrounding words.
2. **Skip-gram**: Predicts the surrounding words given the current word.

#### Example Code using `gensim` for Word2Vec

```python
from gensim.models import Word2Vec

# Sample text corpus
sentences = [["natural", "language", "processing", "is", "fun"],
             ["deep", "learning", "and", "machine", "learning"],
             ["word2vec", "is", "a", "word", "embedding", "technique"]]

# Training Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

# Getting the vector for a word
vector = model.wv['language']
print(vector)  # The word vector for 'language'

# Finding most similar words
print(model.wv.most_similar('language'))
```

### FastText

**FastText** is an extension of Word2Vec developed by Facebook. Unlike Word2Vec, FastText works on subword units (i.e., character n-grams), which allows it to handle rare or misspelled words better.

#### Example Code using `gensim` for FastText

```python
from gensim.models import FastText

# Training FastText model
model = FastText(sentences, vector_size=100, window=5, min_count=1)

# Getting vector for a word
vector = model.wv['language']

# Finding most similar words
print(model.wv.most_similar('language'))
```

### Using Pre-trained Embeddings from Kaggle

Kaggle often provides pre-trained embeddings, such as GloVe or FastText, which can be directly downloaded and used in models. Here's an example of loading pre-trained GloVe embeddings:

```python
import numpy as np

def load_glove_model(file_path):
    glove_model = {}
    with open(file_path, 'r') as file:
        for line in file:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_model[word] = embedding
    return glove_model

# Load the GloVe embeddings (you can find pre-trained files on Kaggle)
glove_model = load_glove_model("glove.6B.100d.txt")

# Example of using GloVe to get the vector of a word
print(glove_model['language'])
```

### Applications of Word Embeddings

1. **Sentiment Analysis**: Word embeddings are used to capture semantic information in text to determine if a sentence expresses positive or negative sentiment.
2. **Document Clustering**: Embeddings allow similar documents to be grouped together by measuring cosine similarity between their word vectors.
3. **Question Answering Systems**: Word embeddings help in identifying semantically related answers to a given question.
4. **Named Entity Recognition (NER)**: Word embeddings can be used to identify entities like people, locations, and organizations in unstructured text.

---

## Conclusion

Both N-Grams and Word Embeddings are essential tools in Natural Language Processing. N-Grams capture sequences of words, while word embeddings map words to dense vectors, encoding their meaning in a way that machines can understand. These techniques serve as foundational blocks for more advanced NLP tasks like machine translation, sentiment analysis, and text classification.

By understanding and implementing these techniques, you can begin building more effective models for a wide range of real-world applications in machine learning and AI.
