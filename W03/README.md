# Week 3: Language modeling and N-grams

Language Modeling and N-grams are two fundamental concepts in the fields of Machine Learning (ML) and Natural Language Processing (NLP).

# Language Modeling
Language modeling is a core task in NLP that involves predicting the next word in a sequence based on the words that have already been observed.

A simple example of a language model is one that uses the probability of word occurrence. For example, given the sentence "It is a nice day", the model could predict the next word by calculating the probability of all words in its vocabulary following this sequence.

Modern language models, such as Transformer models like GPT (Generative Pretrained Transformer) and BERT (Bidirectional Encoder Representations from Transformers), go a step further. Instead of simply calculating word occurrence probabilities, these models understand the context of a sentence by learning the underlying structure of the language.

For instance, the GPT model, when given a prompt like "The weather is nice, I think I will go for a ", might suggest "walk", "run", or "drive", because it has learned from numerous text examples that these words often complete this type of sentence.

# N-grams
N-grams are another fundamental tool in NLP, used to represent text or speech as a collection of n-length word sequences.

For instance, the sentence "I like to play football" can be broken into the following 2-grams (bigrams): ["I like", "like to", "to play", "play football"]. And, into 3-grams (trigrams) as follows: ["I like to", "like to play", "to play football"].

These N-grams can be used as features in many NLP tasks. For example, in sentiment analysis, if the phrase "not good" appears frequently in negative reviews, the model can learn to associate this bigram with a negative sentiment.

By analyzing N-grams, ML models can make more informed decisions about what word or phrase comes next. This enables them to perform tasks like text generation, text classification, and many others.

In conclusion, Language Modeling and N-grams are core techniques in NLP. They enable machines to understand and generate human-like text, powering many applications we use daily, from web search to virtual assistants.

# Readings

[Beyond n-grams: Can linguistic sophistication improve language modeling?](https://aclanthology.org/P98-1028.pdf)

[Show some love to your n-grams: A bit of progress and stronger n-gram language modeling baselines](https://api.repository.cam.ac.uk/server/api/core/bitstreams/83d21f26-066b-4894-915b-63c7749b8a3f/content)

[Building Wikipedia N-grams with Apache Spark
](https://www.researchgate.net/profile/Jorge-Fonseca-10/publication/361805716_Building_Wikipedia_N-grams_with_Apache_Spark/links/63146b815eed5e4bd1468051/Building-Wikipedia-N-grams-with-Apache-Spark.pdf)

Additional article

[Are All Languages Equally Hard to Language-Model?](https://aclanthology.org/N18-2085v2.pdf)

# Code examples

Example 1: Simple Language Model and N-Grams

In this example, we are going to learn about language modeling and N-grams. Language modeling helps us predict the likelihood of a sequence of words occurring in a language. N-grams are groups of N words that appear together in a piece of text.

We will start by creating a simple language model using a short text. The text we have chosen is: "A language model is a type of probabilistic model that assigns probabilities to sequences of words. It is used in various natural language processing tasks."

Next, we will use N-grams to break this text into smaller chunks. For instance, 2-grams (bigrams) will group two words together, like "A language," "language model," and so on. The idea is to find patterns in the text and understand how frequently certain word sequences occur.

Once we have our N-grams, we will build the language model. It will look at each sequence of words and remember which word tends to follow others. For example, after "A language," the word "model" often appears.

Finally, we'll use this language model to generate new text. We'll start with a random set of words from the original text and use the model to predict the next word. Then, we'll keep adding words based on the probabilities learned from the N-grams.

The result will be a new text that looks similar to the original one but may have some variations. It's like a computer trying to write sentences that sound like a human!

Example 2: Complex Language Model and Brown Corpus

In this example, we are going to dive into a more advanced language model using a larger collection of text called the Brown corpus. This corpus contains text from various sources, such as news articles, fiction stories, and more.

First, we'll download and load the Brown corpus using the Natural Language Toolkit (NLTK) library. The NLTK library helps us work with human language data in Python.

Next, instead of using bigrams as before, we will use trigrams (3-grams). Trigrams group three words together, allowing us to capture more complex patterns in the text.

After creating the trigrams, we'll build the language model, just like we did in the previous example. This time, the model will analyze larger chunks of text and learn more about how words connect to each other in different genres.

Finally, we'll use this more powerful language model to generate a longer piece of text, trying to mimic the writing style found in the Brown corpus.

The result will be an extended text that sounds like it was written by an author using similar writing styles to those found in the Brown corpus. This is the kind of thing used in more advanced natural language processing tasks, like chatbots and text generation applications.




