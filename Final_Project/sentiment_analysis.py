import pandas as pd
import numpy as np
import re
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

all_tweets = pd.read_csv("../Final_Project/all_tweets.csv", low_memory=False)
X = all_tweets.tweet
X.dropna(how="any", inplace=True)
X = X.to_frame()

# -------------------------------------------------------------------------------
# adjustments before the sentiment analysis
# -------------------------------------------------------------------------------

# to drop hashtags and usernames in order to perform better sentiment analysis
def find_username_hashtag(string):
    regex = r'[@][^\s#@]+'
    return re.sub(regex, "  ", string)

def find_username_in_row(row):
    return find_username_hashtag(row['tweet'])

X['tweet'] = X.apply(find_username_in_row, axis=1)

# to drop URLs in order to perform better sentiment analysis
def findurl(string):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    return re.sub(regex, "  ", string)

def findurlinrow(row):
    return findurl(row['tweet'])

X['tweet'] = X.apply(findurlinrow, axis=1)

# to replace multiple spaces--thusly URLs, usernames, and hashtags-- with the single space
X['tweet'] = X['tweet'].str.replace('  ', ' ')

# to achive computational efficiency, convert all string to lowercase
X['tweet'] = X['tweet'].str.lower()

# to convert Tweet column of dataframe into a Pandas Series object for the sake of Vectorizer
X = X['tweet'].squeeze()

# -------------------------------------------------------------------------------
# preparation for the sentiment analysis task
# -------------------------------------------------------------------------------

# to convert a bag-of-words vocabulary
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range=(1,2)).fit(X)
X = vect.transform(X)

# to convert sparse matrix to Pandas dataframe to control the content of the matrix
tweets_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())

# Alternatively, Tf-Idf Vectorizer can also be used for this task
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b').fit(X)
# X_tfidf = vectorizer.transform(X)

# -------------------------------------------------------------------------------
# sentiment analysis with three different ML algorithms
# -------------------------------------------------------------------------------

# K-Means Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=44)
k_labels = kmeans.fit_predict(X)
print(f"The score of K-Means Clustering is: {kmeans.score(X)}")

# -------------------------------------------------------------------------------
def convert_to_32bit_indices(x):
    x.indptr = np.array(x.indptr, copy=False, dtype=np.int32)
    x.indices = np.array(x.indices, copy=False, dtype=np.int32)
    return x
X = convert_to_32bit_indices(X)

# kmeans = KMeans(n_clusters=2, random_state=44)
# X_2d = X.reshape(-1, 1)
# kmeans.fit(X_2d)
# labels = kmeans.predict(X_2d)

# In order to visualize K-Means clustering, I wrote the code in the above comment,
# yet, it returned a ValueError:
# "Only sparse matrices with 32-bit integer indices are accepted. Got int64 indices."
# Therefore, I defined convert_to_32bit_indices function, yet it did not work either.
# -------------------------------------------------------------------------------

# Spectral Clustering

# For the Spectral Clustering, I run the convert_to_64bit_indices function, so that
# I get rid of "RuntimeError: nnz of the result is too large" and run Spectral Clustering
# properly. Yet, I cannot make run Spectral Clustering, it did not work.
def convert_to_64bit_indices(x):
    x.indptr = np.array(x.indptr, copy=False, dtype=np.int64)
    x.indices = np.array(x.indices, copy=False, dtype=np.int64)
    return x
X = convert_to_64bit_indices(X)

from sklearn.cluster import SpectralClustering
spectral = SpectralClustering(n_clusters=2, random_state=44, gamma=1.0, n_neighbors=10)
spectral.fit(X)
preds = spectral.predict(X)
print(spectral.affinity_matrix_)

# Gaussian Mixture Model
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2)
gmm_labels = gmm.fit_predict(X.toarray())
proba_lists = gmm.predict_proba(X.toarray())
print(f"The score of Gaussian Mixture model is: {gmm.score(X.toarray())}")
