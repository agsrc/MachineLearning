import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, hinge_loss

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class Perceptron():
    
    def __init__(self, learning_rate=000.00010,epochs=50, add_intercept=True): #learning rate is kept really low
        self.learning_rate=learning_rate
        self.add_intercept = add_intercept
        self.epochs = epochs
        self.fail = []
        self.W = None
    
    def forward_pass(self, X):
        Z = X.dot(self.W.T)
        self.C = sigmoid(Z)
        Z[Z>0] = 1
        Z[Z<=0] = 0                
        return Z
    
    def fit(self, X, Y):
        from sklearn.metrics import accuracy_score
        
        np.random.seed(10)
        
        
        if self.add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        
        if self.W is None:
            self.W = np.random.randn(X.shape[1])
        
        for i in range(self.epochs):
            Z = self.forward_pass(X)
            E = (Y - Z).dot(X)
            self.W = self.W + self.learning_rate * E
            self.fail.append(self.loss(Y, Z))
           
        
    def predict(self, X):
        if self.add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        
        Z = self.forward_pass(X)
        return Z
        
    def score(self, X, Y):
        Z = self.predict(X)
        self.last_preds = Z
        from sklearn.metrics import accuracy_score
        return accuracy_score(Y, Z)
    
    def loss(self, Y, Z):
        return log_loss(Y, Z)
        

from scipy.io import arff
data = arff.loadarff('dataset_8_liver-disorders.arff')



import pandas as pd
df = pd.DataFrame(data[0])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df = pd.DataFrame(df_scaled, columns=df.columns)

Y = df.selector
X = df.drop(columns=['selector'], axis=1)


from sklearn.model_selection import train_test_split

preds =[]

p = Perceptron(learning_rate=.1, epochs=100)
count = []
fail = []
max_count = 0
B_S = None
B_S_confidences = None
B_S_results = None
fail_for_B_S = None
for i in range(1,40):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=i, test_size=.10)
    p.fit(X_train,Y_train)
    #plt.plot(p.fail)
    score=p.score(X_test,Y_test)
    print("Testing score: ", score)
    count.append(score)
    loss=log_loss(Y_test, p.predict(X_test))
    print("Testing loss: ", loss)
    fail.append(loss)
    if (score > max_count):
        max_count = score
        B_S = i
        B_S_confidences = p.C
        B_S_results = Y_test
        B_S_predictions = p.predict(X_test)
        fail_for_B_S = p.fail[len(p.fail)-p.epochs:]
    preds.append(p.last_preds)
    
print("Avg score = ", np.mean(count))
print("Avg  loss = ", np.mean(fail))
print("Best split = ", B_S)
print("Max score = ", max_count)




preview = pd.DataFrame()
preview['actual']=B_S_results
preview['predictions']=B_S_predictions
preview['confidence']=B_S_confidences



plt.plot(fail_for_B_S)




plt.scatter(preview['confidence'],preview['actual'])




from pandas.plotting import scatter_matrix
scatter_matrix(X, figsize=(12,12))


X_new = X





X_new['anotherfeature']=X.sgpt * X.gammagt * 100

scatter_matrix(X_new, figsize=(16,16))


preds =[]

p = Perceptron(learning_rate=.1, epochs=100)
count = []
fail = []
max_count = 0
B_S = None
B_S_confidences = None
B_S_results = None
fail_for_B_S = None
for i in range(1,51):
    X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, random_state=i, test_size=.10)
    p.fit(X_train,Y_train)
    score=p.score(X_test,Y_test)
    print("Testing score: ", score)
    count.append(score)
    loss=log_loss(Y_test, p.predict(X_test))
    print("Testing loss: ", loss)
    fail.append(loss)
    if (score > max_count):
        max_count = score
        B_S = i
        B_S_confidences = p.C
        B_S_results = Y_test
        B_S_predictions = p.predict(X_test)
        fail_for_B_S = p.fail[len(p.fail)-p.epochs:]
    preds.append(p.last_preds)
    
print("Avg score = ", np.mean(count))
print("Avg  loss = ", np.mean(fail))
print("Best split = ", B_S)
print("Max score = ", max_count)




preview = pd.DataFrame()
preview['actual']=B_S_results
preview['predictions']=B_S_predictions
preview['confidence']=B_S_confidences

