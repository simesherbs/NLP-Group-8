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
import time
from scipy.optimize import minimize
from sklearn.multiclass import OneVsRestClassifier
import sys

a =3



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

def create_matrix(csv_file, mode:str, training_hash, weights):
    
    df = pd.read_csv(csv_file)
    if mode == 'train':
        CS_dict = global_CS
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
                CS = 0
                if token in CS_dict[genre].keys():
                    CS_tuple = CS_dict[genre][token]
                    for i in range(0,2):
                        CS += (weights[0] * (CS_tuple[0] + CS_tuple[2])) * (weights[1] * (CS_tuple[1]))
                        CS += CS_dict[genre][token][i] * weights[i] #[0] = atfidf, [1]=chi2, [2]=cval
                sum += CS
            
            CS_mean.append(sum/len(ngrams))
        
        X.loc[len(X)] =  CS_mean
        Y.loc[len(Y)] = Y_tags
    if mode == 'train':
        return X, Y, CS_dict
    return X,Y

    
def run_system(weights):
    
    training_X, training_Y, training_hash = create_matrix(train_csv, 'train', {}, weights)
    testing_X, testing_Y = create_matrix(test_csv, 'test', training_hash, weights)


    scaler = preprocessing.MinMaxScaler()
    #training_X = pd.DataFrame(scaler.fit_transform(training_X), index=training_X.index, columns=training_X.columns)
    #testing_X = pd.DataFrame(scaler.fit_transform(testing_X), index=testing_X.index, columns=testing_X.columns)



    lr = SVC()

    ovr = OneVsRestClassifier(lr)

    # Train the classifier
    ovr.fit(training_X, training_Y)
    y_pred = ovr.predict(testing_X)

    from sklearn.metrics import classification_report

    # Print the classification report (Precision, Recall, F1-score for each genre)
    report = classification_report(testing_Y, y_pred, target_names=classes, zero_division=np.nan, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    report_df.to_csv('classifier_runs/' + str(weights) +'.csv')
    print(report['macro avg']["f1-score"])
    return report['macro avg']["f1-score"]


            
        

seed=42
 # Rows = documents, Columns = num_bins
train_csv, test_csv, classes = split_corpus('english_movies.csv', seed)
global_CS = calc_CS(train_csv)

def objective_function(weights):
    start_time = time.time()
    # Run the system with the current set of weights
    f1_score = run_system(weights)  # Assume run_system returns F1 score
    end_time = time.time()
    run_time.append(end_time-start_time)
    return -f1_score  # Minimize negative F1 score for maximization

# Example: Searching for weights between 0.1 and 1.0 in increments of 0.1
run_time = []
import numpy as np
import itertools

# Define the features
features = ["feature1", "feature2", "feature3"]
step_size = 0.1  # Step size for weight increments

# Generate all possible weight combinations
possible_weights = np.arange(0, 1 + step_size, step_size)
all_combinations = itertools.product(possible_weights, repeat=len(features))

# Filter combinations where weights sum to 1.0
valid_combinations = [comb for comb in all_combinations if np.isclose(sum(comb), 1.0)]

# Example function to evaluate the system
def evaluate_system(weights):
    print(f"Evaluating with weights: {weights}")
    return objective_function(weights)  # Replace with your actual evaluation function

# Evaluate each valid combination
best_score = 0
best_weights = None
for weights in valid_combinations:
    score = evaluate_system(weights)
    if score > best_score:
        best_score = score
        best_weights = weights

print(f"Best Weights: {best_weights}, Best Score: {best_score}")


print("average run time: ", sum(run_time)/len(run_time))
print("elapsed run time: ", sum(run_time))

