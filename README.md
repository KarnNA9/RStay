# Week 2: Text Preprocessing and Tokenization

Week 2 of the course focuses on Text Preprocessing and Tokenization, fundamental techniques used in Natural Language Processing (NLP). In this week, students will learn how to prepare textual data for analysis by performing essential preprocessing steps, such as lowercasing and punctuation removal. Additionally, they will explore the concept of tokenization, which involves breaking text into individual units (tokens) for further analysis. A practical exercise will guide students in implementing a simple tokenizer to gain hands-on experience.

## Readings:

[What is a word, What is a sentence? Problems of Tokenization](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=aa80793d1d41d5241017a8fc755b7efc362a6439)

[Text preprocessing for text mining in organizational research](https://journals.sagepub.com/doi/pdf/10.1177/1094428120971683)

[Text preprocessing for unsupervised learning](https://arthurspirling.org/documents/preprocessing.pdf)

## Text Preprocessing and Its Importance:
Text preprocessing is a crucial step in NLP, aimed at transforming raw text data into a more structured and manageable format. It involves various operations, such as converting text to lowercase, removing punctuation, handling special characters, and eliminating irrelevant information like stop words. The primary goals of text preprocessing are to reduce the dimensionality of the data, improve data quality, and enhance the efficiency and accuracy of NLP models. Clean and standardized text data enable more meaningful analysis, better model training, and facilitate more accurate predictions.

## Tokens and Tokenization:
Tokens are the atomic units of text, representing individual elements such as words or characters in a sentence. Tokenization is the process of breaking down text into these discrete units. Word-level tokenization splits text into words, while character-level tokenization breaks text into individual characters. Tokenization is the foundation for various NLP tasks, including language modeling, sentiment analysis, and machine translation. Proper tokenization helps capture the inherent structure of natural language and allows algorithms to process text efficiently.

## Features in the Context of Machine Learning:
In machine learning, features are the measurable characteristics or attributes extracted from data that represent the input to a predictive model. For NLP tasks, features can include word frequencies, n-grams, and other statistical representations derived from the tokenized text. Defining relevant features is essential because they capture the relevant information from the data that can influence the final prediction. Carefully selected features help the machine learning model understand the underlying patterns in the text and make accurate predictions for tasks like text classification, named entity recognition, and sentiment analysis.

## Practical exercise: Implement a simple tokenizer

The provided code is a Python script that allows users to input the URL of a webpage they wish to parse for textual content. The script uses the requests library to fetch the webpage content and then employs BeautifulSoup from the bs4 library to parse the HTML and extract the text content from the webpage. The extracted text is then saved to a text file named parsed_text.txt. After saving the content to the file, the script proceeds with text preprocessing, converting the text to lowercase and removing punctuation using regular expressions. Next, it tokenizes the preprocessed text into individual words and characters. The resulting word and character tokens are printed to the console, allowing users to observe the parsed content in a structured and processed form. This code is useful for gathering and preprocessing textual data from a webpage, which can be further analyzed or used in natural language processing tasks.

    python -m venv venv
    source ./venv/bin/activate
    pip install -r requirements.txt
    python main.py
