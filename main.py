# confusion matrix to find model accuracy
from sklearn.metrics import confusion_matrix, classification_report
#  data visualization library based on matplotlib
import seaborn as sn
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
# for normalization of the datset later
from sklearn import preprocessing
import statsmodels.api as sm
import scipy.optimize as opt
import numpy as np
import pylab as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import cohen_kappa_score, jaccard_score
# dataset
chd_data = pd.read_csv("massachusetts.csv")
chd_data.drop(['education'], inplace=True, axis=1)


# removing NaN / NULL values
chd_data.dropna(axis=0, inplace=True)
print(chd_data.head(), chd_data.shape)
print(chd_data.TenYearCHD.value_counts())

# counting no. of patients affected with CHD
plt.figure(figsize=(7, 5))
sn.countplot(x='TenYearCHD', data=chd_data, palette="BuGn_r")
plt.show()

# Training and Test Sets: Splitting Data | Normalization of the Dataset

X = np.asarray(chd_data[['age', 'male', 'cigsPerDay', 'totChol', 'glucose']])
y = np.asarray(chd_data['TenYearCHD'])


# //////////////////////////////////////////////////////////
# 3 normalization of the datset - which is only on the x axis
X = preprocessing.StandardScaler().fit(X).transform(X)

# Train-and-Test -Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

# //////////////////////////////////////////////////////////
# 4 Modeling of the Dataset | Evaluation and Accuracy :
# Here we're going to use the Jaccard Similarity Coefficient or simply the jaccard score
# And it is simply a statistic used for estimating the similarity and diversity of sample sets
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Evaluation and accuracy
print('')
print('Accuracy of the model in jaccard score is = ',
      jaccard_score(y_test, y_pred))

# 5 Using Confusion Matrix to find the Acuuracy of the model :
# Confusion matrix
# table that is often used to describe the performance of a classification model (or "classifier") on a   datatest for which the true values are known.

cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data=cm,
                           columns=['Predicted:0', 'Predicted:1'],
                           index=['Actual:0', 'Actual:1'])
plt.figure(figsize=(8, 5))
sn.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")
plt.show()

print('The details for confusion matrix is =')
print(classification_report(y_test, y_pred))
