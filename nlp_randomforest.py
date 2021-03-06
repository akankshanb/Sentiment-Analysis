#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 19:34:29 2017

@author: ubuntu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
#Importing dataset
dataset = pd.read_csv('Restaurant_Reviews.csv', delimiter = '\t' , quoting = 3 )
# delimiter is tab
# we need to ensure the double quote does not create problem. code is 3 

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder
y = dataset.iloc[:,1].values #dependent variable to see the review as pos or neg
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y) 

#cleaning the text
#BOW Model will extract relevant words from the reviews
#stemming- love verb abd loved- only one will be taken

import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


corpus = [] #empty list of all the reviews
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ' ,  dataset['Review'][i]) #for 1st review i = 0
     #letters only taken
    
    
    review = review.lower()
    #convert all letters into lowercase
    
    #to remove insignificant words - like the that an in etc.
    #import nltk
    #nltk.download('stopwords')
    #from nltk.corpus import stopwords
    review = review.split() #as the string of review is broken into words
    #review = [word for word in review if not word in set(stopwords.words('english'))] 
    
    #STEMMING - to avoid sparsity
    #from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    # for all the words 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#BOW MOdel
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # keep 1500 words from 1565
# to create sparse matrix i.e take unique words as columns
X = cv.fit_transform(corpus).toarray() #sparse matrix


#CLASSIFICATON MODEL
#RANDOM FOREST

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=19, criterion='entropy', random_state=0 )
classifier.fit(X_train, y_train)
#n_estimators = number of trees - detect overfitting

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
standard_deviation = accuracies.std()
count = 0
for value in y_pred:
    if value == 1:
        count = count +1
    else:
        continue
    
   
p1 = float(count)/len(y_pred)
entropy1 = -p1*math.log(p1,2) 
p2 = float(len(y_pred) - count)/len(y_pred)
entropy2 = -p2*math.log(p2,2) 

Accuracy = (cm[0][0] + cm[1][1])/200.0
#Accuracy = (TP + TN) / (TP + TN + FP + FN)

#Precision = TP / (TP + FP)
Precision = float(cm[1][1])/(cm[1][1]+ cm[1][0])              
#Recall = TP / (TP + FN)
Recall = float(cm[1][1])/(cm[1][1]+cm[0][1])

F1_Score = 2 * Precision * Recall / (Precision + Recall)

