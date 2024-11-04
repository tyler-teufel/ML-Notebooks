# Work Embeddings + Words as Vectors

## Overview

### Word Embeddings

**Word Embeddings** represent words as numbers, specifically as numeric vectors, in a way that captures semantic relationships and contextual information.



**What does that actually mean?**

Words with simiular meanings are positioned clsoer to eachother and the distance and direction between vectors encode the degree of similarity between words.

Word Embeddingss have become an essential tool in **Natural Language Processing (NLP)**, For tasks like text classification, Named Entity Recognition (NER), word similarity / analogy tasks, and Q + A, to name a few.

**Common Type of Embeddings**:
  * Frequency-based
    * TF-IDF
  * Prediction-Based
  * Contextual Based 


**Word2vec**: Two main architectures, Continuous Bag of Words (CBOW), and Skip-gram.
  * CBOW: Predict the target words given a context word.
  * Skip-Gram: Predict the context words given a target word.

**GloVe**: Global Vectors for Word Representation
    * Used co-occurence statistics to create models
    * Takes a broader view, by analyzing how often words appear together across entire corpus. 
