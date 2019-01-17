import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler 

import time
import timeit




start = timeit.default_timer()
t0 = time.clock()
#file read
DS = pd.read_csv('diabetes.csv')

#preprocessing --replacing zeroes
pd.DataFrame.hist(DS, figsize=[20,20])
noZero= ['Glucose', 'BloodPressure','SkinThickness','BMI','Insulin']
classifiers=[]
y_pred = []
acc =[]
max=-1

for C in noZero:
    DS[C]= DS[C].replace(0, np.NaN)
    mean=int(DS[C].mean(skipna=True))
    DS[C]=DS[C].replace(np.NaN,mean)

	#splitting DataSet
X = np.array(DS.drop(['Outcome'],1))

Y = np.array(DS['Outcome'])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0,test_size=0.2)

for x in range(1,61):
    clf = neighbors.KNeighborsClassifier(n_neighbors = x, n_jobs = -1, algorithm = 'auto')
    classifiers.append(clf)
    clf_model_1 = clf.fit(X_train, Y_train)
    yPred= cross_val_predict(clf_model_1, X, Y, cv=10)
    y_pred.append(yPred)
    score = accuracy_score(Y, yPred)
    acc.append(score)
    if max < acc[x-1]:
        max = acc[x-1]
        position = x
    print('Accuracy with k = %f is %f'%(x, acc[x-1]))
    
plt.plot(range(1,61),acc, color='red', marker='o', linestyle='dashed',linewidth=3, markersize=12)
plt.xlabel('K-Neighbor')
plt.ylabel('Accuracy')
plt.savefig('diabetics.pdf')
plt.show(block=True)

print('Max Accuracy is with k = %f and Accuracy is %f'%(position, acc[position-1]))

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
                     metric_params=None, n_jobs=1,n_neighbors=11,p=2,
                     weights='uniform')

#predicting the test results
Y_pred =cross_val_predict(classifier,X,Y,cv=10)

#evaluating-- accuracy
cm=confusion_matrix(Y,Y_pred)
print(cm)


stop = timeit.default_timer()
print('Time: ', stop - start)  
print (time.clock() - t0, "seconds")