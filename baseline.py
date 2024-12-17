from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC

import pandas as pd
from bigramstrigrams import generate_ngram_file
import time
start = time.time()
def split_corpus(corpusfile, seed):
    parsed  = generate_ngram_file(corpusfile)
    corpus_df = pd.read_csv(parsed)
    X = corpus_df['Ngrams']
    Y = corpus_df['Genres']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=seed)
 
    training = pd.DataFrame({
        'Ngrams': X_train,
        'Genres': y_train
    })
    training["Genres"] = training["Genres"].apply(
        lambda x: x.split(", ")
    )
    train = "train.csv"
    training.to_csv(train)
    testing = pd.DataFrame({
        'Ngrams': X_test,
        'Genres': y_test
    })
    testing["Genres"] = testing["Genres"].apply(
        lambda x: x.split(", ")
    )
    test = "test.csv"
    testing.to_csv(test)
    corpus_df["Genres"] = corpus_df["Genres"].apply(
        lambda x: x.split(", ")
    )
    unique_genres = set(genre for genre_list in corpus_df['Genres'] for genre in genre_list)
    genres_arr = sorted(unique_genres)
    return training, testing, genres_arr



seed=42
 # Rows = documents, Columns = num_bins
train_df, test_df, classes = split_corpus('english_movies.csv', seed)







vocabulary = []

for id, row in train_df.iterrows():

    temp = row['Ngrams'].split(', ')
    vocabulary += temp[0:len(temp)-1]



def tokenizer(doc):
    return str(doc).split(', ')
vocabulary = set(vocabulary)
# Convert multi-label genres to binary arrays
mlb = MultiLabelBinarizer(classes=classes)
vectorizer = TfidfVectorizer(vocabulary=vocabulary, token_pattern=None, tokenizer=tokenizer)
X_train = vectorizer.fit_transform(train_df['Ngrams'])
X_test = vectorizer.fit_transform(test_df['Ngrams'])
y_train = mlb.fit_transform(train_df['Genres'])
y_test = mlb.fit_transform(test_df['Genres'])

# Use Logistic Regression with a OneVsRest strategy for multi-label classification
classifier = OneVsRestClassifier(SVC())
classifier.fit(X_train, y_train)


# Predict on the test set
y_pred = classifier.predict(X_test)
end = time.time()
# Print classification report

print(f"{end-start:.2} seconds") 
print(classification_report(y_test, y_pred, target_names=mlb.classes_))



