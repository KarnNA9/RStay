# Week 11: Transfer learning for emotion detection

Transfer Learning
Transfer learning is a technique in machine learning where a pre-trained model is adapted for a different but related task. Instead of starting the learning process from scratch, the model leverages the knowledge gained during training on a related task.

How does Transfer Learning Work?
The process involves taking a pre-trained model, often trained on a large-scale task such as image classification or language translation, and fine-tuning this model on a specific task. The fine-tuning step involves continuing the training process on the new task, allowing the model to adapt to the specific features of the new task while maintaining the general knowledge learned from the initial task.

Transfer learning has proven to be highly successful, especially in deep learning tasks, because it allows for the training of high-performance models even when only small amounts of labeled data are available for the specific task at hand.

Emotion Detection in Text
Emotion detection in text, also a part of the broader field of Natural Language Processing (NLP), aims at identifying and classifying the emotional tone conveyed in a source of text. It goes beyond the binary positive-negative classification of sentiment analysis, aiming to classify text into multiple emotion categories like joy, sadness, anger, surprise, etc.

Emotion detection is a complex task due to the inherent ambiguity and variability in expressing emotions in text. Factors such as context, culture, and language nuances like sarcasm and irony add to this complexity.

Transfer Learning for Emotion Detection
In the context of emotion detection, transfer learning can be particularly beneficial. State-of-the-art NLP models like BERT, GPT, or RoBERTa, which are pre-trained on large text corpora, can serve as a starting point. These models have learned a rich understanding of language structure and semantics, which can be effectively leveraged for emotion detection.

The process is as follows:

Pre-training: Models like BERT are pre-trained on a large corpus of text, learning to predict the next word in a sentence (language model objective) or to tell if two sentences are in order (sentence prediction objective). During this process, they learn rich word representations and capture the syntax and semantics of the language.

Fine-tuning: The pre-trained model is then fine-tuned on an emotion detection task. During fine-tuning, the model learns to associate the text's emotional tone with its existing understanding of language structure and semantics.

Transfer learning allows the model to take advantage of the general language understanding learned from the large text corpus and combine it with the specific task of recognizing emotions. This approach typically leads to superior performance compared to training a model from scratch, particularly when the amount of labeled data for the emotion detection task is limited.

# Readings

[Transfer learning](https://ftp.cs.wisc.edu/machine-learning/shavlik-group/torrey.handbook09.pdf)


[Emotion detection in text: a review](https://arxiv.org/pdf/1806.00674.pdf)

# Code examples

# Transfer learning

# transfer.py

Step 1: Data Preprocessing

We start by importing the necessary libraries, including PyTorch and torchvision. We define data transformations using torchvision.transforms.Compose. These transformations include resizing the images to (224, 224) pixels, converting the images to tensors, and normalizing the pixel values to a range of [-1, 1]. We then load the CIFAR-10 dataset and create DataLoader for training and testing, enabling us to efficiently load batches of data during training.

Step 2: Loading the Pre-trained Model

Next, we load a pre-trained ResNet18 model from torchvision.models. ResNet18 is a popular convolutional neural network architecture that has been pre-trained on a large dataset (e.g., ImageNet). This pre-training enables the model to capture meaningful features from images effectively.

Step 3: Modifying the Model for CIFAR-10

Since CIFAR-10 has 10 classes, we need to modify the last fully connected layer of the pre-trained ResNet18 to have 10 output units (one for each class). We achieve this by replacing the last layer with a new fully connected layer. Additionally, we freeze all the other layers in the pre-trained model, meaning we won't update their weights during training. This ensures that we retain the pre-trained features that the model has learned.

Step 4: Moving the Model to GPU

We check if a GPU is available and move the model to the GPU for faster computation if it is available. Otherwise, the model will be moved to the CPU.

Step 5: Training the Model

We define the loss function (CrossEntropyLoss) and the optimizer (SGD) for training. We then run a loop for a specified number of epochs to train the model. Inside the loop, we set the model to training mode using model.train(), and then iterate through the training data batches. For each batch, we perform the forward pass, compute the loss, and update the model's parameters through backpropagation and optimization using model.backward() and optimizer.step().

Step 6: Testing the Model

After training, we evaluate the model's performance on the test dataset. We set the model to evaluation mode using model.eval() to disable certain operations like dropout that are active during training but not during evaluation. We then iterate through the test data batches, make predictions using the trained model, and calculate the accuracy by comparing the predicted labels with the ground truth labels.

Step 7: Displaying Results

Finally, we display the training loss for each epoch during training and the accuracy on the test set after training.

# Text emotion detection 

# ted.py

Step 1: Loading the Pre-trained BERT Model and Tokenizer

We start by importing the necessary libraries, including torch for PyTorch and the transformers library by Hugging Face. The transformers library provides pre-trained language models like BERT. We load the pre-trained BERT model (bert-base-uncased) and the corresponding tokenizer. The tokenizer helps us convert raw text into a format suitable for BERT.

Step 2: Defining Emotion Labels

Next, we define a dictionary of emotion labels. Since we are working with an emotion detection model, we have four emotion categories: "happy," "sad," "angry," and "neutral." The emotion labels will be used to interpret the model's output.

Step 3: The detect_emotion Function

We define a function called detect_emotion that takes a text as input and aims to detect the emotion conveyed by the text. Inside the function, we use the pre-trained BERT tokenizer to convert the input text into tensors, suitable for BERT model input. We also handle text truncation and padding to ensure uniform input size.

Step 4: Using the Pre-trained BERT Model

Within the detect_emotion function, we pass the tokenized text through the pre-trained BERT model. The model is a BERT-based sequence classification model that has been fine-tuned specifically for emotion detection. It outputs logits, which are unnormalized values representing the model's confidence scores for each emotion category.

Step 5: Getting the Detected Emotion

We convert the logits into probabilities using the softmax function, which scales the logits to represent probabilities for each emotion class. We then select the emotion label with the highest probability as the detected emotion for the input text.

Step 6: Testing the Model

We test the model's performance using some sample texts. We call the detect_emotion function for each text and print the original text along with the detected emotion for each sample.

# ted_go_emotions.py

Step 1: Importing Libraries

We start by importing the necessary libraries for our text emotion detection task. These libraries include pandas for handling data, torch for deep learning, transformers for BERT-based models, and requests for downloading the dataset.

Step 2: Downloading and Joining the Go Emotions Dataset

In this step, we define a function download_goemotions_dataset to download and join the Go Emotions dataset. We provide three URLs to download three parts of the dataset and concatenate them into a single DataFrame. The function ensures that the dataset is downloaded and joined only if it's not already present in the data/full_dataset directory.

Step 3: Loading the Go Emotions Dataset

Now that the dataset is downloaded and joined (or if it already existed), we load it into a pandas DataFrame. The dataset contains text samples labeled with multiple emotions.

Step 4: Preprocessing the Dataset

We extract the text samples and emotion labels from the DataFrame. The text samples will be our input, and the emotion labels will be our target for training the emotion detection model.

Step 5: Defining the BERT Model and Tokenizer

In this step, we define the BERT model and its tokenizer using the BertTokenizer and BertForSequenceClassification classes from the transformers library. The model is a BERT-based sequence classification model, fine-tuned for emotion detection, and the tokenizer helps convert raw text into suitable input for the BERT model.

Step 6: Creating a Custom Dataset

We create a custom dataset class named GoEmotionsDataset to handle our preprocessed data. This dataset class will help us efficiently feed the data into the deep learning model during training and testing.

Step 7: Training the Emotion Detection Model (Omitted)

The code contains a commented section that would represent the training loop. The actual training is omitted here for brevity.

Step 8: Emotion Detection Results

After training the model (omitted for this example), we proceed to demonstrate emotion detection on some example sentences. We define four example sentences to test the model's capability to detect emotions in text.

Step 9: Using the Trained Model for Emotion Detection

For each example sentence, we use the trained model to predict the associated emotions. We preprocess the sentence using the BERT tokenizer and feed it into the model. The model outputs logits, which are then transformed into probabilities using the sigmoid function.

Step 10: Displaying Emotion Detection Results

Finally, we display the emotion detection results for each example sentence. The detected emotions are the ones with probabilities above 0.5 (chosen as the threshold). We print the original sentence and the emotions detected by the model.