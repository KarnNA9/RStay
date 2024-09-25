# Week 7: Word Embeddings in NLP Applications
Word embeddings are a type of word representation that allows words with similar meanings to have similar representations. They are a key breakthrough in the field of Natural Language Processing (NLP).

The fundamental idea behind word embeddings is to convert words or phrases from the vocabulary into vectors of real numbers, which can be used as input to various NLP tasks. These vectors capture a wealth of information, such as semantic and syntactic meaning, relation with other words, etc.

How do Word Embeddings Work?
Word embeddings are learned using neural networks, either as part of a larger task or with the specific goal of mapping words onto vectors. The neural network's hidden layers capture the input word's context and represent the word as a dense vector. The position of a word within the vector space is learned from text and is based on the words that surround the word when it is used.

Two popular models for generating word embeddings are Word2Vec, developed by researchers at Google, and GloVe (Global Vectors for Word Representation), developed by researchers at Stanford. Both models take a corpus of text as input and produce a vector space, with each unique word in the corpus being assigned a corresponding vector in the space.

Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located close to one another in the space. This means semantically similar words are effectively mapped to nearby points.

Applications of Word Embeddings
Word embeddings are used in many NLP tasks because they provide a dense representation of words and their relative meanings. They can be used in:

Text Classification: Word embeddings can be used as feature vectors for raw text to feed into machine learning algorithms.

Text Similarity: The cosine of the angle between word vectors provides an effective method for measuring word similarity.

Named Entity Recognition: Word embeddings can be used to capture context in text and help in identifying named entities.

Sentiment Analysis: They can also be used to understand the sentiment expressed in a given body of text, like movie or product reviews.

Machine Translation: Word embeddings can be used to translate from one language to another.

For instance, in sentiment analysis, where the goal is to determine the sentiment of a text passage, word embeddings can help capture the meaning of the words and their context, which can significantly improve the model's performance.


# Readings

[Word Embeddings: A Survey](https://arxiv.org/pdf/1901.09069.pdf)


[A Survey of Word Embeddings Evaluation Methods](https://arxiv.org/pdf/1801.09536.pdf)


[Evaluation methods for unsupervised word embeddings](https://aclanthology.org/D15-1036.pdf)

Additional Article

[Word Embedding Evaluation and Combination](https://aclanthology.org/L16-1046/)

# Code examples

Example 1: Simple Word Embeddings

In this example, we introduce the concept of Word Embeddings using the Word2Vec model with Gensim. Word embeddings are dense vector representations that capture semantic relationships between words, enabling NLP models to understand the meaning and context of words.

Data Preparation:
We start with a collection of 20 simple sentences, each containing 8 to 10 words. These sentences cover various NLP-related topics, such as Natural Language Processing, Machine Learning, Text Classification, and more.

Building Word2Vec Model:
Using Gensim's Word2Vec class, we create a Word2Vec model with the given sentences. We set the vector size to 10, which means each word will be represented as a 10-dimensional vector in the embedding space. A window of 3 indicates that the model will consider a maximum of 3 words before and after the target word while learning the embeddings.

Obtaining Word Embeddings:
After training the Word2Vec model, we can access the word embeddings using the model.wv attribute. We demonstrate retrieving the embedding vector for the word 'NLP'.

Finding Similar Words:
We also explore the model's ability to find similar words to a given word based on cosine similarity. In this example, we find words similar to 'NLP'.

Example 2: Complex Word Embeddings (Pre-trained Model)

In this example, we showcase the use of a pre-trained Word2Vec model from Gensim's gensim.downloader module. Pre-trained models have been trained on large-scale datasets and can capture complex word relationships and meanings.

Loading Pre-trained Model:
We load the 'word2vec-google-news-300' model, which contains word embeddings trained on a vast corpus of Google News articles. The model's dimensionality is 300, meaning each word is represented as a 300-dimensional vector.

Word Embeddings and Similar Words:
Similar to Example 1, we obtain the word embedding for the word 'cat' from the pre-trained model and find the most similar words based on cosine similarity.
