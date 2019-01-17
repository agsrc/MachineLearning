import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score

df = pd.read_csv('adult.csv')
#df = df.drop(['Id'],axis=1)
target = df['income']
# display(df)
df = df.select_dtypes(['number'])

data=pd.read_csv("adult.csv")

data = data[(data != '?').all(1)]

x = LabelEncoder()

#transforimng discrete features
workclass_param = x.fit_transform(data['workclass'])
education_param = x.fit_transform(data['education'])
martial_param = x.fit_transform(data['marital.status'])
occupation_param = x.fit_transform(data['occupation'])
relationship_param = x.fit_transform(data['relationship'])
race_param = x.fit_transform(data['race'])
native_param = x.fit_transform(data['native.country'])
income_param = x.fit_transform(data['income'])
sex_param = x.fit_transform(data['sex'])

#creating new dataframe post dealing with discrete features
data_new = data[['age', 'fnlwgt', 'capital.gain', 'capital.loss', 'hours.per.week']].copy()

data_new['workclass'] = workclass_param
data_new['education'] = education_param
data_new['marital.status'] = martial_param
data_new['occupation'] = occupation_param
data_new['relationship'] = relationship_param
data_new['race'] = race_param
data_new['native.country'] = native_param
data_new['income'] = income_param
data_new['sex'] = sex_param

features = list(set(data_new.columns) - set(['income']))

cat_subset = pd.get_dummies(data, columns=['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'native.country'])
cat_subset.drop('education.num', axis =1)

sex_category = x.fit_transform(cat_subset['sex'])
income_category = x.fit_transform(data['income'])

cat_subset.drop('sex', axis =1)
cat_subset.drop('income', axis =1)

cat_subset['sex'] = sex_category
income_category[income_category==0]=-1
cat_subset['income'] = income_category
    
X = cat_subset.drop(['income'],axis=1)
Y = cat_subset['income']


x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print(y_train.shape)
print(x_train.shape)
print(y_test.shape)
print(x_test.shape)

'''


(836,)
(836, 86)
(93,)
(93, 86)
'''
def apply_kernel(kernel, X):
    if kernel != None:
        global X_TRAIN
        newX = X
        f1=[]
        for i in range(len(newX)):
            f1.append(kernel(newX[i], X_TRAIN))
        f1 = np.array(f1)
        f1 = f1.reshape(newX.shape[0], -1)
        return np.hstack((newX, f1))
    else: return X
	
	X_TRAIN = None
KERNEL = None

def linear(x, y):
    out_vec = []
    x = np.array(x).reshape(1, x.shape[0])
    y = np.array(y).T
    return  1e-04 * x.dot(y)

def poly(x, y, degree = 2):
    out_vec = []
    x = np.array(x).reshape(1, x.shape[0])
    y = np.array(y).T
    if degree == None:
        degree = y.shape[0]
    return 1e-07 * ((x.dot(y)+1) ** degree)

def rbf(x, y):
    out_vec = []
    x = np.array(x).reshape(1, x.shape[0])
    y = np.array(y)
    for i in range(len(y)):
        out_vec.append(np.exp(-(1/1e04)*(x - y[i])**2))
    return out_vec

def svm(X, Y, epochs, kernel=linear, learning_rate=0.001):
    global X_TRAIN, KERNEL
    rows = X.shape[0]
    X_TRAIN = X
    KERNEL = kernel
    X = apply_kernel(kernel, X)
    np.random.seed(23)
    W = np.random.normal(0,0.05, X.shape[1])
    columns = X.shape[1]
    for e in range(1, epochs + 1):
        yHat = X.dot(W)
        output = yHat * Y
        for i in range(len(output)):
            o = output[i]
            if o > 1:
                W = W - learning_rate * (1/e) * W
            else:
                xty = X[i].T * Y[i]
                W = W + learning_rate * (xty - ((1/e) * W))
    return np.array(W)
	weights = svm(np.array(x_train), y_train, 1000, kernel=rbf)
# print(weights)
def predict(X, W):
    y = []
    rows = X.shape[0]
    X = apply_kernel(KERNEL, X)
    columns = X.shape[1]
    yHat = X.dot(W)
    y = np.array(yHat)
    y[y>0]=1
    y[y<=0]=-1
    return y
	from sklearn.metrics import accuracy_score
predictions = predict(x_test, weights)
print(accuracy_score(y_test, predictions))
print(f1_score(y_test, predictions, average='weighted'))
print(recall_score(y_test, predictions, average='weighted'))
print(explained_variance_score(y_test, predictions, multioutput='uniform_average'))
'''
0.7849462365591398
0.6903744008291228
0.7849462365591398
0.7795863423888448
'''
weights1 = svm(np.array(x_train), y_train, 1000, kernel=linear)


predictions1 = predict(x_test, weights1)
print(accuracy_score(y_test, predictions1))
print(f1_score(y_test, predictions1, average='weighted'))
print(recall_score(y_test, predictions1, average='weighted'))
print(explained_variance_score(y_test, predictions1, multioutput='uniform_average'))

'''
0.7849462365591398
0.6903744008291228
0.7849462365591398
0.7493829464236728
'''
weights2 = svm(np.array(x_train), y_train, 1000, kernel=poly)

predictions2 = predict(x_test, weights2)
print(accuracy_score(y_test, predictions2))
print(f1_score(y_test, predictions2, average='weighted'))
print(recall_score(y_test, predictions2, average='weighted'))
print(explained_variance_score(y_test, predictions2, multioutput='uniform_average'))

'''


0.6451612903225806
0.6732331937946319
0.6451612903225806
-0.9479452054794524
'''