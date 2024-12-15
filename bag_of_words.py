import hashlib
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer
from bigramstrigrams import generate_ngram_file
from final_helpers import get_vocab
from final import calc_CS
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing as preprocessing
from sklearn.multiclass import OneVsRestClassifier
import sys

corpus = sys.argv[1]
weights = [.3,.3,.3]

def split_corpus(corpusfile):
    parsed  = generate_ngram_file(corpusfile)
    corpus_df = pd.read_csv(parsed)
    X = corpus_df['Ngrams']
    Y = corpus_df['Genres']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
 
    training = pd.DataFrame({
        'Ngrams': X_train,
        'Genres': y_train
    })
    train = "train.csv"
    training.to_csv(train)
    testing = pd.DataFrame({
        'Ngrams': X_test,
        'Genres': y_test
    })
    test = "test.csv"
    testing.to_csv(test)
    corpus_df["Genres"] = corpus_df["Genres"].apply(
        lambda x: x.split(", ")
    )
    unique_genres = set(genre for genre_list in corpus_df['Genres'] for genre in genre_list)
    genres_arr = sorted(unique_genres)
    return train, test, genres_arr

def create_matrix(csv_file, mode:str, training_hash):
    
    df = pd.read_csv(csv_file)
    if mode == 'train':
        CS_dict = calc_CS(csv_file, weights=weights)
    else:
        CS_dict = training_hash
    df["Genres"] = df["Genres"].apply(
        lambda x: x.split(", ")
    )
    unique_genres = set(genre for genre_list in df['Genres'] for genre in genre_list)
    genres_arr = sorted(unique_genres)
  
    columns =  genres_arr 

    
    X = pd.DataFrame(columns=columns)
    Y = pd.DataFrame(columns=genres_arr)
    for index, row in df.iterrows():
        CS_mean = []
        Y_tags = []
        ngrams = row['Ngrams']
        genres = row['Genres']
        ngrams = str(ngrams).split(', ')
        for genre in genres_arr:
            sum = 0
            if genre in genres:
                Y_tags.append(1)
            else:
                Y_tags.append(0)
            for token in ngrams:
                if token in CS_dict[genre].keys():
            
                    sum += CS_dict[genre][token]
                    
            
            CS_mean.append(sum/len(ngrams))
        
        X.loc[len(X)] =  CS_mean
        Y.loc[len(Y)] = Y_tags
    if mode == 'train':
        return X, Y, CS_dict
    return X,Y

    

train_csv, test_csv, classes = split_corpus('english_movies.csv')
training_X, training_Y, training_hash = create_matrix(train_csv, 'train', {})
testing_X, testing_Y = create_matrix(test_csv, 'test', training_hash)


scaler = preprocessing.MinMaxScaler()
#training_X = pd.DataFrame(scaler.fit_transform(training_X), index=training_X.index, columns=training_X.columns)
#testing_X = pd.DataFrame(scaler.fit_transform(testing_X), index=testing_X.index, columns=testing_X.columns)



lr = SVC()

ovr = OneVsRestClassifier(lr)

# Train the classifier
ovr.fit(training_X, training_Y)
y_pred = ovr.predict(testing_X)
print(testing_X)
print(testing_Y)
print(y_pred)
from sklearn.metrics import classification_report

# Print the classification report (Precision, Recall, F1-score for each genre)
report = classification_report(testing_Y, y_pred, target_names=classes, zero_division=np.nan, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classifier_runs/' + str(weights))


            
        


 # Rows = documents, Columns = num_bins
