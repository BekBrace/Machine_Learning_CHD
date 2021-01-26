from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm
import scipy.optimize as opt
import numpy as np
import pylab as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score

# Dataset
chd_data = pd.read_csv('massachusetts.csv')
chd_data.drop(['education'], inplace=True, axis=1)

# Removing NaN
chd_data.dropna(axis=0, inplace=True)
# print(chd_data.head(), chd_data.shape)
# print(chd_data.TenYearCHD.value_counts())

# Counting no. of patients affected with CHD

plt.figure(figsize=(8, 6))
sn.countplot(x="TenYearCHD", data=chd_data, palette="BuGn_r")
# plt.show()

# Train and test sets
# ----------------------
# Declaration of x and y variables (axis)
x = np.asarray(chd_data[['age', 'male', 'cigsPerDay', 'totChol', 'glucose']])
y = np.asarray(chd_data['TenYearCHD'])

# Normalize the dataset
x = preprocessing.StandardScaler().fit(x).transform(x)

# Actually train and test x and y sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=4)
print('Train set: ', x_train.shape, y_train.shape)
print('Test set: ', x_test.shape, y_test.shape)

# Modeling the dataset
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)

# Evaluation and accuracy
print('')
print('Accuracy of the model in Jaccard score is : ',
      jaccard_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_setup = pd.DataFrame(data=cm, columns=[
                        'Preddicted:0', 'Preddicted:1'], index=['Actual:0', 'Actual:1'])
plt.figure(figsize=(9, 6))
sn.heatmap(cm_setup, annot=True, fmt='d', cmap="Greens")
plt.show()

print('The details for confusion matrix is : ')
print(classification_report(y_test, y_pred))
