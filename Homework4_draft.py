import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

imm = pd.read_csv("../Homework4/immSurvey.csv")
X, y = imm.text, imm.sentiment

# to convert a bag-of-words vocabulary
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
vect = CountVectorizer(ngram_range=(2,2), stop_words=ENGLISH_STOP_WORDS).fit(X)
X = vect.transform(X)

# to check the content of the sparse matrix as dataframe
X_array = X.toarray()
X_df = pd.DataFrame(X_array, columns=vect.get_feature_names())

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=44)

# support vector regression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
svr = SVR()
grid_svr = GridSearchCV(svr, {'C':[0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 10]})
grid_svr.fit(X_train, y_train)
grid_svr.predict(X_test)
print("Best CV parameter for C value in SVR: \n", grid_svr.best_params_)
print("Accuracy on training data for SVR: \n", grid_svr.score(X_train, y_train))
print("Accuracy on test data for SVR: \n", grid_svr.score(X_test, y_test))

# random forest regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000, random_state=44)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Accuracy on training data for RF: \n", rf.score(X_train, y_train))
print("Accuracy on test data for RF: \n", rf.score(X_test, y_test))

# TfIdf - term frequency inverse document frequency
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(2, 2)).fit(imm.text)
X_tfidf = vectorizer.transform(imm.text)