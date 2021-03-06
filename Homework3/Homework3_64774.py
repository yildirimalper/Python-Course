import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cses = pd.read_csv("../Homework3/cses4_cut.csv")

# Extracting target variable from the dataframe
y = cses['voted'].to_numpy()
y = y.astype(float)

# Extracting educational level, employment type, and socio economic status as explanatory variables
X1 = cses[['D2012', 'D2013']].to_numpy()
X1 = X1.astype(float)
for i in range(6,10):
	X1[X1==i] = np.nan
X2 = cses[['D2003']].to_numpy()
X2 = X2.astype(float)
for i in range(96,100):
	X2[X2==i] = np.nan

# Imputation of data
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X1 = imp.fit_transform(X1)
X2 = imp.fit_transform(X2)
X = np.concatenate((X1, X2), axis=1)

# Scaling Explanatory Variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# K-NEAREST NEIGHBORS CLASSIFICATION

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print(f"""The accuracy score of k-NN Classifier given the number of neighbors is: {accuracy_score(y_test, knn_pred)}
The 10-fold cross-validation output is: \n {cross_val_score(knn, X_train, y_train, cv=10)}
Classification Report for k-NN Classifier is: \n {classification_report(y_test, knn_pred)}""")

# Visualization of k-NN results for different n_neighbors
neighbors = np.arange(1,9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train, y_train)
	train_accuracy[i] = knn.score(X_train, y_train)
	test_accuracy[i] = knn.score(X_test, y_test)

plt.title("k-NN: Different Number of Neighbors")
plt.plot(neighbors, test_accuracy, color="royalblue", label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, color="firebrick", label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

# Confusion Matrix for k-NN Classification
# In order to observe the misclassification of the model, confusion matrix is applied.
cf_knn = confusion_matrix(y_test, knn_pred)
sns.heatmap(cf_knn/np.sum(cf_knn), square=True, annot=True, fmt='.2%', cbar=False, cmap='RdBu') # visualized as percentages
plt.title('Confusion Matrix of k-NN Classification')
plt.xlabel('Predicted Value')
plt.ylabel('True Value')
plt.show()

#??GAUSSIAN NAIVE BAYES CLASSIFICATION

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB(var_smoothing=2e-9)
gnb_pred = gnb.fit(X_train, y_train).predict(X_test)
print(f"""The accuracy score of Gaussian Naive Bayes is: {accuracy_score(y_test, gnb_pred)}
The 5-fold cross-validation output is: \n {cross_val_score(gnb, X_train, y_train, cv=5)}""")

# Confusion Matrix for Gaussian Naive Bayes
cf_gnb = confusion_matrix(y_test, gnb_pred)
sns.heatmap(cf_gnb/np.sum(cf_gnb), square=True, annot=True, fmt='.2%', cbar=False, cmap='Blues')
plt.title('Confusion Matrix of GaussianNB')
plt.xlabel('Predicted Value')
plt.ylabel('True Value')
plt.show()

# LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
logreg = LogisticRegression()
grid_logreg = GridSearchCV(logreg, {'C':[0.001, 0.01, 0.1, 1, 10, 100]}, cv=10)
y_pred = grid_logreg.fit(X_train, y_train).predict(X_test)
print(f"""The accuracy score of Logistic Regression is: {accuracy_score(y_test, y_pred)}
Best regularization (C) value for Logistic Regression is: {grid_logreg.best_params_}
Classification Report for Logistic Regression is: \n {classification_report(y_test, y_pred)}""")

# Confusion Matrix for Logistic Regression
cf_logreg = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_logreg/np.sum(cf_logreg), square=True, annot=True, fmt='.2%', cbar=False, cmap='Greens')
plt.title("Confusion Matrix for Logistic Regression")
plt.xlabel("Predicted Value")
plt.ylabel("True Value")
plt.show()