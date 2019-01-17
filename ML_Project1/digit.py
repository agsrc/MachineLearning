import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def procedure():
    time.sleep(2.5)

t0=time.time()
DS = pd.read_csv('train.csv')

#splitting DataSet

C_drop = []
for column in DS.columns:
    if DS[column].sum() == 0:
        C_drop.append( column)

print("dropping: ", len(C_drop), "columns")

X = np.array(DS.drop(['label', *C_drop],1))
X=DS.iloc[:, 1:783]
Y=DS.iloc[:,0]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0,test_size=0.2)

#Feature scaling- when distance and normalization is involved

scale_x= StandardScaler()
X_train=scale_x.fit_transform(X_train)
X_test=scale_x.transform(X_test)

#defining the model using KNN
#Y_test=sqrt(12.4)--12-1 taking odd to classify accurately

classifier=KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')

#Fit model

classifier.fit(X_train, Y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=-1,n_neighbors=11,p=2,
                     weights='uniform')

#predicting the test results using cross validation

Y_pred =cross_val_predict(classifier,X,Y,cv=10)

# confusion matrix

cm=confusion_matrix(Y,Y_pred)
print(cm)

#evaluating--f1_score and accuracy
print("F1 score is: ", f1_score(Y, Y_pred))
print("accuracy is :", (accuracy_score(Y,Y_pred)*100),"%")
print (time.time()-t0)