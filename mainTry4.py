import os
import requests
import json_lines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from googletrans import Translator
from langdetect import detect
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import WhitespaceTokenizer, word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, f1_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier

import nltk
nltk.download('punkt')
nltk.download('stopwords')

X=[]; y=[]; z=[]
with open ("reviews.jl","rb") as f:
    for item in json_lines.reader(f):
        X.append(item["text"])
        y.append(item["voted_up"])
        z.append(item["early_access"])
fields = ['text', 'voted_up', 'early_access']
obj = {}
obj[fields[0]] = X
obj[fields[1]] = y
obj[fields[2]] = z

df = pd.DataFrame(columns=['text','text_translate','voted_up','early_access'])
df['text'] = pd.Series(X)
df['voted_up'] = pd.Series(y)
df['early_access'] = pd.Series(z)

#remove non-english
i = 0 
X_ENG = {} 
y_ENG = {} 
z_ENG = {} 
Obj_ENG = {} 
for idx, text in enumerate(X):
    try:
        det = detect(text)
        if det == 'en':
            X_ENG[idx] = text
            y_ENG[idx] = y[idx]
            z_ENG[idx] = z[idx]
    except:
        print('IS GIBBERISH')
Obj_ENG[fields[0]] = X_ENG
Obj_ENG[fields[1]] = y_ENG
Obj_ENG[fields[2]] = z_ENG

df_eng = pd.DataFrame(columns=['text','voted_up','early_access'])
df_eng['text'] = pd.Series(X_ENG)
df_eng['voted_up'] = pd.Series(y_ENG)
df_eng['early_access'] = pd.Series(z_ENG)

# GET TARGET
target = df_eng['early_access']
print(target)
print(df_eng['text'])

#Logisteic Regression part in Q1

max_df_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] 
# max_df default is 1.0
# so will remove words that appear in more than 100% of docs
min_df_values = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.80, 0.90, 0.99] 
# min_df, default 1, will remove words that appear in less than 1 doc
# in both cases default removes none
#iterate through the min and max values by turn, manually

ci_range= [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 1000.0]
mean_range = []
std_range = []

for ci in ci_range: #iterate through the min and max values by turn, manually
    tokenizer = WhitespaceTokenizer().tokenize
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=nltk.corpus.stopwords.words('english'), max_df = 0.7, min_df = 1, norm=None)
    X_fit = vectorizer.fit_transform(df_eng['text'])
    df_fit = pd.DataFrame(X_fit.toarray(),columns=vectorizer.get_feature_names())

    temp = []

    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(df_fit, target, test_size=0.2, random_state=i )

        # Run logistic regression 
        logisticReg = LogisticRegression(C = ci, penalty= 'l1', solver='liblinear')
        logisticReg.fit(X_train, y_train)

        predictions = logisticReg.predict(X_test)
        #print(confusion_matrix(y_test, predictions))
        #print(classification_report(y_test, predictions))
    
        scores = logisticReg.predict_proba(X_test)
        fpr, tpr, _= roc_curve(y_test, scores[:, 1])
        print('AUC = {}'.format(auc(fpr, tpr)))
        temp.append(format(auc(fpr, tpr)))
    mean_range.append(np.array(temp).astype(np.float).mean())
    std_range.append(np.array(temp).astype(np.float).std())
plt.errorbar(ci_range, mean_range, yerr=std_range)
plt.xlabel('ci'); plt.ylabel('Mean AUC score')
plt.title('Range of AUC scores and their varience with respect to C values')
plt.show()

    
#Q1 KNN part
neighbours = [1, 2, 3, 4, 5, 10, 50, 100]
for neigh in neighbours: #iterate through the K value in KNN
    tokenizer = WhitespaceTokenizer().tokenize
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=nltk.corpus.stopwords.words('english'), max_df = 0.7, min_df = 1, norm=None)
    X_fit = vectorizer.fit_transform(df_eng['text'])
    df_fit = pd.DataFrame(X_fit.toarray(),columns=vectorizer.get_feature_names())

    temp = []

    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(df_fit, target, test_size=0.2, random_state=i )

        # Run logistic regression 
        model = KNeighborsClassifier(n_neighbors= neigh, weights= 'uniform')
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        #print(confusion_matrix(y_test, predictions))
        #print(classification_report(y_test, predictions))
    
        scores = model.predict_proba(X_test)
        fpr, tpr, _= roc_curve(y_test, scores[:, 1])
        print('AUC = {}'.format(auc(fpr, tpr)))
        temp.append(format(auc(fpr, tpr)))
    mean_range.append(np.array(temp).astype(np.float).mean())
    std_range.append(np.array(temp).astype(np.float).std())
plt.errorbar(neighbours, mean_range, yerr=std_range)
plt.xlabel('K'); plt.ylabel('Mean AUC score')
plt.title('Range of AUC scores and their varience with respect to K values')
plt.show()