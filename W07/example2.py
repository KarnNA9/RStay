import gensim.downloader as api

# Load pre-trained Word2Vec model
model = api.load("word2vec-google-news-300")

# Get word vector for a specific word
word_vector = model['cat']

# Find most similar words to 'cat'
similar_words = model.most_similar('cat', topn=5)

print("Word Vector for 'cat':")
print(word_vector)

print("\nMost Similar Words to 'cat':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.2f}")
