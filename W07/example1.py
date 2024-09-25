from gensim.models import Word2Vec

# Simple example sentences
sentences = [
    ['I', 'love', 'NLP', 'and', 'machine', 'learning', 'techniques', 'for', 'language', 'processing'],
    ['Word', 'embeddings', 'are', 'powerful', 'for', 'NLP', 'tasks', 'like', 'sentiment', 'analysis'],
    ['NLP', 'is', 'fascinating', 'and', 'opens', 'a', 'world', 'of', 'possibilities'],
    ['Machine', 'learning', 'can', 'be', 'applied', 'to', 'various', 'NLP', 'applications'],
    ['Natural', 'Language', 'Processing', 'involves', 'processing', 'human', 'language', 'data', 'for', 'computers'],
    ['Text', 'classification', 'is', 'an', 'important', 'NLP', 'task', 'for', 'categorizing', 'documents'],
    ['Word', 'embeddings', 'represent', 'words', 'in', 'a', 'continuous', 'vector', 'space'],
    ['Word', 'similarity', 'can', 'be', 'measured', 'using', 'cosine', 'similarity'],
    ['Language', 'models', 'help', 'to', 'generate', 'text', 'and', 'understand', 'context'],
    ['Sequence', 'to', 'sequence', 'models', 'are', 'used', 'for', 'machine', 'translation'],
    ['Deep', 'learning', 'has', 'revolutionized', 'NLP', 'in', 'recent', 'years'],
    ['Chatbots', 'are', 'an', 'example', 'of', 'conversational', 'AI', 'systems'],
    ['Named', 'Entity', 'Recognition', 'identifies', 'entities', 'in', 'a', 'text'],
    ['Word', 'embeddings', 'capture', 'semantic', 'relationships', 'between', 'words'],
    ['POS', 'tagging', 'labels', 'parts', 'of', 'speech', 'in', 'a', 'sentence'],
    ['Sentiment', 'analysis', 'determines', 'the', 'sentiment', 'of', 'a', 'text', 'as', 'positive', 'or', 'negative'],
    ['Machine', 'translation', 'translates', 'text', 'from', 'one', 'language', 'to', 'another'],
    ['Text', 'summarization', 'generates', 'a', 'concise', 'summary', 'of', 'a', 'text'],
    ['Named', 'Entity', 'Recognition', 'is', 'important', 'for', 'information', 'extraction'],
    ['Text', 'generation', 'can', 'be', 'achieved', 'using', 'language', 'models']
]

# Create Word2Vec model
model = Word2Vec(sentences, vector_size=10, window=3, min_count=1, workers=4)

# Get word embeddings
word_embeddings = model.wv

# Get word vector for a specific word
word_vector = word_embeddings['NLP']

# Find most similar words to 'NLP'
similar_words = word_embeddings.most_similar('NLP')

print("Word Vector for 'NLP':")
print(word_vector)

print("\nMost Similar Words to 'NLP':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.2f}")
