BreadcrumbsSpam-Message-Detection-NLP-Project-

In this project, I learned Natural Language Processing (NLP) techniques to build a robust spam message detection system. Here’s a breakdown of what I used and how it works:

1. Data Preprocessing: I started by importing the necessary libraries including spacy for NLP processing and pandas for data manipulation. The dataset was loaded from a CSV file (train.csv) containing SMS messages, which was then preprocessed.

2. Text Preprocessing: The SMS messages underwent essential preprocessing steps:
Lemmatization: Each word in the messages was converted to its base form using spaCy's lemmatization capabilities to normalize the text.
Stopword Removal: Common words (stopwords) that don't contribute to the meaning of the message were removed using spaCy's built-in stopwords list.

3. Machine Learning Pipeline: To train the model, I utilized a pipeline from scikit-learn:
TF-IDF Vectorization: The text data was transformed into numerical TF-IDF (Term Frequency-Inverse Document Frequency) vectors, which weigh the importance of each word in the messages.
Random Forest Classifier: Chosen for its ability to handle high-dimensional data and its effectiveness in text classification tasks.

4. Model Training and Evaluation: The dataset was split into training and testing sets using train_test_split. The model was then trained on the training data and evaluated on the test data, achieving an accuracy score of 98% on unseen data.

5. Performance Metrics: To assess the model’s performance, I employed:
Classification Report: Providing precision, recall, F1-score, and support metrics for each class (spam and non-spam).
Confusion Matrix: Visualized using seaborn, depicting true positives, false positives, true negatives, and false negatives.
