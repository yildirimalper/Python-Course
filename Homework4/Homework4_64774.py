import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import re

imm = pd.read_csv("../Homework4/immSurvey.csv")
X, y = imm.text, imm.sentiment

# to convert a bag-of-words vocabulary
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range=(2,2)).fit(X)
X = vect.transform(X)

# to check the content of the sparse matrix as dataframe
X_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())

# scaling the target variable
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
y = y.values.reshape(-1, 1) # otherwise, StandardScaler returns an error
y = scaler.fit_transform(y)
y = y.flatten() # otherwise, SVR returns an error

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=44)

# support vector regression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
svr = SVR()
grid_svr = GridSearchCV(svr, {'C':[0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 10]})
grid_svr.fit(X_train.toarray(), y_train)
svr_pred = grid_svr.predict(X_test.toarray())
print("Best CV parameter for C value in SVR: \n", grid_svr.best_params_)
print("Correlation matrix for test group and predictions in SVR: \n", np.corrcoef(y_test, svr_pred))

# random forest regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000, random_state=44)
rf.fit(X_train.toarray(), y_train)
rf_pred = rf.predict(X_test.toarray())
print("Correlation matrix for test group and predictions in RF: \n", np.corrcoef(y_test, rf_pred))

# Gaussian process regressor
from sklearn.gaussian_process import GaussianProcessRegressor
gpr = GaussianProcessRegressor(normalize_y=False)
gpr.fit(X_train.toarray(), y_train)
gpr_pred = gpr.predict(X_test.toarray())
print("Correlation matrix for test group and predictions in GaussianPR: \n", np.corrcoef(y_test, gpr_pred))

# TfIdf - term frequency inverse document frequency
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b').fit(imm.text)
X_tfidf = vectorizer.transform(imm.text)

# train test split again
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.5, random_state=44)

# SVR after Tf-Idf Transformation
tfidf_svr = SVR()
tfidf_svr.fit(X_train.toarray(), y_train)
tfidf_pred = tfidf_svr.predict(X_test.toarray())
print("Correlation matrix for test group and predictions in SVR after Tf-Idf: \n", np.corrcoef(y_test, tfidf_pred))

# RF after Tf-Idf Transformation
tfidf_rf = RandomForestRegressor(n_estimators=1000, random_state=44)
tfidf_rf.fit(X_train.toarray(), y_train)
tfidf_pred_rf = tfidf_rf.predict(X_test.toarray())
print("Correlation matrix for test group and predictions in RF after Tf-Idf: \n", np.corrcoef(y_test, tfidf_pred_rf))

# GaussianPR after Tf-Idf Transformation
gpr_tfidf = GaussianProcessRegressor(normalize_y=False)
gpr_tfidf.fit(X_train.toarray(), y_train)
tfidf_pred_gpr = gpr_tfidf.predict(X_test.toarray())
print("Correlation matrix for test group and predictions in GaussianPR: \n", np.corrcoef(y_test, tfidf_pred_gpr))

# Correlation coefficients for only bigrams

# SVR with CountVectorizer = 0.510
# RF with CountVectorizer = 0.420
# GaussianPR with CountVectorizer = 0.370
# SVR with TfIdfVectorizer = 0.300
# RF with TfIdfVectorizer = 0.343
# GaussianPR with TfIdfVectorizer = 0.395

# Correlation coefficient for uni- and bigrams

# SVR with CountVectorizer = 0.684
# RF with CountVectorizer = 0.693
# GaussianPR with CountVectorizer = 0.317
# SVR with TfIdfVectorizer = 0.503
# RF with TfIdfVectorizer = 0.706
# GaussianPR with TfIdfVectorizer = 0.571

# exploratory visualization for the report
plt.hist(imm['sentiment'], bins=15, color="darkorange", edgecolor="black")
plt.title("Sentiment Distribution of Dataset")
plt.xlabel("Sentiment Score")
plt.ylabel("Counts")
plt.show()
plt.savefig("sentiment_distribution.png")