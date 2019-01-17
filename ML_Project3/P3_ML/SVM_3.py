# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 11:06:23 2018

@author: Akshay
"""
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, hinge_loss
from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix

import seaborn as sns
sns.set()


#read the csv
data=pd.read_csv("adult.csv")

#Abandon samples with missing terms: There are several rows of data containing ' ?'. Abandon the
#rows that contain ' ?'.
data = data[(data != '?').all(1)]

#The first requirement is that all data should be numerical. 
#Therefore, if you have categorical features, they need to be converted to numerical values using 
#variable transformation techniques like one-hot-encoding, label-encoding etc.

#workclass = ['Private', 'State-gov', 'Federal-gov', 'Self-emp-not-inc']
workclass = np.unique(data['workclass'])

#education is numbered as education.num
#education = ['1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'HS-grad', 'Masters', 'Prof-school', 'Some-college']
#martial = ['Divorced', 'Married-civ-spouse', 'Never-married', 'Separated', 'Widowed']
#occupation = ['Adm-clerical', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving']
#relationship = ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']
#race = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']
#sex = ['Female', 'Male']
#native_country = ['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia']
#income = ['<=50K', '>50K']

education = np.unique(data['education'])
martial = np.unique(data['marital.status'])
occupation = np.unique(data['occupation'])
relationship = np.unique(data['relationship'])
race = np.unique(data['race'])
sex = np.unique(data['sex'])
native = np.unique(data['native.country'])
income = np.unique(data['income'])
hours = np.unique(data['hours.per.week']) # Not discrete

#enc = preprocessing.OrdinalEncoder(handle_unknown='ignore')
#categories=[workclass, martial, occupation, relationship, race, sex,native_country, income]
#enc.fit(data)

#enc.fit(data)

x = LabelEncoder()

#transforimng discrete features
workclass_labels = x.fit_transform(data['workclass'])
education_labels = x.fit_transform(data['education'])
martial_labels = x.fit_transform(data['marital.status'])
occupation_labels = x.fit_transform(data['occupation'])
relationship_labels = x.fit_transform(data['relationship'])
race_labels = x.fit_transform(data['race'])
native_labels = x.fit_transform(data['native.country'])
income_labels = x.fit_transform(data['income'])
sex_labels = x.fit_transform(data['sex'])

#creating new dataframe post dealing with discrete features
data_new = data[['age', 'fnlwgt', 'capital.gain', 'capital.loss', 'hours.per.week']].copy()

data_new['workclass'] = workclass_labels
data_new['education'] = education_labels
data_new['marital.status'] = martial_labels
data_new['occupation'] = occupation_labels
data_new['relationship'] = relationship_labels
data_new['race'] = race_labels
data_new['native.country'] = native_labels
data_new['income'] = income_labels
data_new['sex'] = sex_labels

#spliting into Features and Targets
X = data_new.drop(columns=['income'], axis=1)
y = data_new['income']

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#scater plot and analysis

features = list(set(data_new.columns) - set(['income']))


#Function to create Train and Test set from the original dataset 

def getTrainTestData(dataset,split):
    np.random.seed(0) 
    training = [] 
    testing = []
    np.random.shuffle(dataset) 
    shape = np.shape(dataset)
    trainlength = np.uint16(np.floor(split*shape[0]))
    for i in range(trainlength):
        training.append(dataset[i])
    for i in range(trainlength,shape[0]): 
        testing.append(dataset[i])
    training = np.array(training) 
    testing = np.array(testing)
    return training,testing

#Function to evaluate model performance. it will take the predicted and actual output as input to calculate the percentage accuracy

def getAccuracy(pre,ytest): 
    count = 0
    for i in range(len(ytest)):
        if ytest[i]==pre[i]: 
            count+=1
    acc = float(count)/len(ytest)
    return acc

#Create Training and Testing data for performance evaluation
#Xtrain, Xtest = getTrainTestData(X, 0.7)

#ytrain, ytest = getTrainTestData(y, 0.7)

#shape = np.shape(Xtrain)

#print("Shape of the dataset ",shape)



def entropy(target_col):
   
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

total_entropy = entropy(data_new['income'])

def InfoGain(data,split_attribute_name,target_name="income"):
    
    #Calculate the entropy of the total dataset
    #total_entropy = entropy(data[target_name])
    
    ##Calculate the entropy of the dataset
    
    #Calculate the values and the corresponding counts for the split attribute 
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    
    #Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    #Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

ig_age = InfoGain(data_new, 'age', 'income')
ig_workclass = InfoGain(data_new, 'workclass', 'income')
ig_fnlwgt = InfoGain(data_new, 'fnlwgt', 'income')
ig_education = InfoGain(data_new, 'education', 'income')
ig_martial = InfoGain(data_new, 'marital.status', 'income')
ig_occupation = InfoGain(data_new, 'occupation', 'income')
ig_relationship = InfoGain(data_new, 'relationship', 'income')
ig_sex = InfoGain(data_new, 'sex', 'income')
ig_race = InfoGain(data_new, 'race', 'income')
ig_gain = InfoGain(data_new, 'capital.gain', 'income')
ig_loss = InfoGain(data_new, 'capital.loss', 'income')
ig_hours = InfoGain(data_new, 'hours.per.week', 'income')
ig_native = InfoGain(data_new, 'native.country', 'income')

print("Information Gain of age is : ", ig_age)
print("Information Gain of workclass is : ", ig_workclass)
print("Information Gain of fnlwgt is : ", ig_fnlwgt)
print("Information Gain of education is : ", ig_education)
print("Information Gain of marital.status is : ", ig_martial)
print("Information Gain of occupation is : ", ig_occupation)
print("Information Gain of relationship is : ", ig_relationship)
print("Information Gain of sex is : ", ig_sex)
print("Information Gain of race is : ", ig_race)
print("Information Gain of capital.gain is : ", ig_gain)
print("Information Gain of capital.loss is : ", ig_loss)
print("Information Gain of hours.per.week is : ", ig_hours)
print("Information Gain of native.country is : ", ig_native)

#total_ig = ig_age+ig_workclass+ig_fnlwgt+ig_education+ig_martial+ig_occupation+ig_relationship+ig_sex+ig_race+ig_gain+ig_loss+ig_hours+ig_native



#scater plot and analysis

features = list(set(data_new.columns) - set(['income']))

# Calculate and plot
corr_matrix = data_new[features].corr()
sns.heatmap(corr_matrix)

# pairplot may become very slow with the SVG format
#%config InlineBackend.figure_format = 'png' 
sns.pairplot(data_new[features]);

plt.scatter(data_new['fnlwgt'], data_new['relationship'])


categorical_subset = pd.get_dummies(data, columns=['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'native.country'])
categorical_subset.drop('education.num', axis =1)

sex_category = x.fit_transform(categorical_subset['sex'])
income_category = x.fit_transform(data['income'])

categorical_subset.drop('sex', axis =1)
categorical_subset.drop('income', axis =1)

categorical_subset['sex'] = sex_category
income_category[income_category==0]=-1
categorical_subset['income'] = income_category

categorical_subset.head()



class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors yi(xi.w+b) = 1
        

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001,
                      ]

        
        
        # extremely expensive
        b_range_multiple = 1
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 1
        latest_optimum = self.max_feature_value*10
        w_t_best = np.array([latest_optimum,latest_optimum])
        b_best = -1
        count = 0
        w_t = 0
        count_min = math.inf
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            print("w is :",w)
            # we can do this because convex
            optimized = False
            
            while not optimized:
                #print("b range is ", self.max_feature_value*b_range_multiple)
                #print("w is ",w)
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        #print("IN tranform matrix")
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        # 
                        # #### add a break here later..
                        #outer = True
                        count = 0
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                 
                        if(count <=   count_min):
                            w_t_best = w_t
                            b_best = b
                            count_min = count
                            print("1. count ", count)

                        if found_option:
                            print("found option is true")
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step
                    
            opt_dict[np.linalg.norm(w_t)] = [w_t_best,b_best]
            print("OPT Dict",opt_dict)
            print("count min ",count_min)
            norms = sorted([n for n in opt_dict])
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
            
            #self.w = w_t_best
            #self.b = b_best
            #latest_optimum = w_t_best[0] + step*2
            count_min = 300000
            
        for i in self.data:
            for xi in self.data[i]:
                yi=i
                #print(xi,':',yi*(np.dot(self.w,xi)+self.b))            

    def predict(self,features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        #if classification !=0 and self.visualization:
        #    self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self, data_dict):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

        plt.show()
        

categorical_subset_2features = categorical_subset[['fnlwgt']].copy()

categorical_subset_2features['reationship'] = relationship_labels
X = categorical_subset_2features.as_matrix()

#categorical_subset_2features.reset_index(drop=True, inplace=True)



d = {}
#for i in categorical_subset['income'].unique():
#    d[i] = [[categorical_subset_2features['fnlwgt'][j], categorical_subset_2features['marital'][j]] for j in categorical_subset[categorical_subset['income']==i].index]
#    d[i] = np.asarray(d[i], dtype=np.int32)

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, random_state=None)
# X is the feature set and y is the target
#skf.get_n_splits(X, y)
#print(skf)  

accuracy = 0
count_predict = 0
svm1 = Support_Vector_Machine()
y = categorical_subset['income'].as_matrix()
for train_index, test_index in skf.split(X,y):
    categorical_subset_2features_train_1 = X[train_index]
    categorical_subset_2features_train = pd.DataFrame({'fnlwgt':categorical_subset_2features_train_1[:,0],'reationship':categorical_subset_2features_train_1[:,1]})
    y_train_folds_1 = y[train_index]
    y_train_folds = pd.DataFrame({'income':y_train_folds_1[:,]})
    categorical_subset_2features_test_1 = X[test_index]
    categorical_subset_2features_test = pd.DataFrame({'fnlwgt':categorical_subset_2features_test_1[:,0],'reationship':categorical_subset_2features_test_1[:,1]})
    y_test_folds_1 = y[test_index]
    y_test_folds = pd.DataFrame({'income':y_test_folds_1[:,]})
    for i in y_train_folds.income.unique():
        d[i] = [[categorical_subset_2features_train['fnlwgt'][j], categorical_subset_2features_train['reationship'][j]] for j in y_train_folds[y_train_folds['income']==i].index]
        d[i] = np.asarray(d[i], dtype=np.int32)
    
    svm1.fit(data=d)
    
    features_test_split = categorical_subset_2features_test
    
    features_predict_test = svm1.predict(features_test_split)
    count_predict = 0
    for p in features_predict_test:
        if features_predict_test[p] == y_test_folds_1[p]:
            count_predict = count_predict + 1
    accuracy = count_predict / len(features_predict_test) + accuracy
    

total_accuracy = accuracy/10    
svm1.visualize(d)

class Support_Vector_Machine_all_features:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors yi(xi.w+b) = 1
        

        step_sizes = [self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001,
                      # point of expense:
                      self.max_feature_value * 0.0001,
                      ]

        
        
        # extremely expensive
        b_range_multiple = 100
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 10
        latest_optimum = self.max_feature_value*10
        w_t_best = np.array([latest_optimum,latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum])
        b_best = -1
        count = 0
        count_min = 300000
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum, latest_optimum])
            print("w is :",w)
            # we can do this because convex
            optimized = False
            while not optimized:
                #print("b range is ", self.max_feature_value*b_range_multiple)
                #print("w is ",w)
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        # 
                        # #### add a break here later..
                        outer = True
                        count = 0
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    outer = False
                                    count = count + 1
                                    break
                            if not outer:
                                break
                                    #print(xi,':',yi*(np.dot(w_t,xi)+b))
                        if(count < count_min):
                            w_t_best = w_t
                            b_best = b
                            count_min = count

                        if found_option:
                            print("found option is true")
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step
                    
            opt_dict[np.linalg.norm(w_t)] = [w_t_best,b_best]
            print("OPT Dict",opt_dict)
            print("count min ",count_min)
            norms = sorted([n for n in opt_dict])
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
            
            #self.w = w_t_best
            #self.b = b_best
            #latest_optimum = w_t_best[0] + step*2
            count_min = 300000
            
        for i in self.data:
            for xi in self.data[i]:
                yi=i
                #print(xi,':',yi*(np.dot(self.w,xi)+self.b))            

    def predict(self,features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0], features[1], features[2], features[3], features[4], features[5], features[6], features[7], features[8], features[9], features[10], features[11], features[12], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self, data_dict):
        [[self.ax.scatter(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

        plt.show()

        

categorical_subset_allfeatures = categorical_subset[['age', 'fnlwgt', 'capital.gain', 'capital.loss', 'hours.per.week', 'sex', 'income']].copy()

categorical_subset_allfeatures['marital'] = martial_labels
categorical_subset_allfeatures['education'] = education_labels
categorical_subset_allfeatures['occupation'] = occupation_labels
categorical_subset_allfeatures['relationship'] = relationship_labels
categorical_subset_allfeatures['race'] = race_labels
categorical_subset_allfeatures['native'] = native_labels
categorical_subset_allfeatures['workclass'] = workclass_labels


categorical_subset_allfeatures.reset_index(drop=True, inplace=True)

d_all = {}
for i in categorical_subset_allfeatures['income'].unique():
    d_all[i] = [[categorical_subset_allfeatures['age'][j], categorical_subset_allfeatures['fnlwgt'][j], categorical_subset_allfeatures['capital.gain'][j], categorical_subset_allfeatures['capital.loss'][j], categorical_subset_allfeatures['hours.per.week'][j], categorical_subset_allfeatures['sex'][j], categorical_subset_allfeatures['marital'][j], categorical_subset_allfeatures['education'][j], categorical_subset_allfeatures['occupation'][j], categorical_subset_allfeatures['relationship'][j], categorical_subset_allfeatures['race'][j], categorical_subset_allfeatures['native'][j], categorical_subset_allfeatures['workclass'][j]]  for j in categorical_subset_allfeatures[categorical_subset_allfeatures['income']==i].index]
    d_all[i] = np.asarray(d_all[i], dtype=np.int32)

skf_all = StratifiedKFold(n_splits=10, random_state=None)
# X is the feature set and y is the target
#skf_all.get_n_splits(X, y)
#print(skf)  


accuracy_all = 0
count_predict_all = 0
d_all = {}
svm1 = Support_Vector_Machine_all_features()
y = categorical_subset['income'].as_matrix()
#categorical_subset_allfeatures = categorical_subset_allfeatures.drop('income')
for train_index, test_index in skf.split(X,y):
    categorical_subset_2features_train_1 = X[train_index]
    categorical_subset_2features_train = pd.DataFrame({'fnlwgt':categorical_subset_2features_train_1[:,0],'reationship':categorical_subset_2features_train_1[:,1]})
    y_train_folds_1 = y[train_index]
    y_train_folds = pd.DataFrame({'income':y_train_folds_1[:,]})
    categorical_subset_2features_test_1 = X[test_index]
    categorical_subset_2features_test = pd.DataFrame({'fnlwgt':categorical_subset_2features_test_1[:,0],'reationship':categorical_subset_2features_test_1[:,1]})
    y_test_folds_1 = y[test_index]
    y_test_folds = pd.DataFrame({'income':y_test_folds_1[:,]})
    for i in y_train_folds.income.unique():
        d[i] = [[categorical_subset_2features_train['fnlwgt'][j], categorical_subset_2features_train['reationship'][j]] for j in y_train_folds[y_train_folds['income']==i].index]
        d[i] = np.asarray(d[i], dtype=np.int32)
    
    svm1.fit(data=d)
    
    features_test_split = categorical_subset_2features_test
    
    features_predict_test = svm1.predict(features_test_split)
    count_predict = 0
    for p in features_predict_test:
        if features_predict_test[p] == y_test_folds_1[p]:
            count_predict = count_predict + 1
    accuracy = count_predict / len(features_predict_test) + accuracy
    

total_accuracy = accuracy/10    
svm1.visualize(d)



accuracy_all = 0
count_predict_all = 0
svm1_all = Support_Vector_Machine_all_features()
d_all = {}
for train_index, test_index in skf.split(categorical_subset_allfeatures): 
    print("Train:", train_index, "Test:", test_index) 
    categorical_subset_allfeatures_train, categorical_subset_allfeatures_test = categorical_subset_allfeatures[train_index], categorical_subset_allfeatures[test_index] 
    for i in categorical_subset_allfeatures['income'].unique():
        d_all[i] = [[categorical_subset_allfeatures['age'][j], categorical_subset_allfeatures['fnlwgt'][j], categorical_subset_allfeatures['capital.gain'][j], categorical_subset_allfeatures['capital.loss'][j], categorical_subset_allfeatures['hours.per.week'][j], categorical_subset_allfeatures['sex'][j], categorical_subset_allfeatures['marital'][j], categorical_subset_allfeatures['education'][j], categorical_subset_allfeatures['occupation'][j], categorical_subset_allfeatures['relationship'][j], categorical_subset_allfeatures['race'][j], categorical_subset_allfeatures['native'][j], categorical_subset_allfeatures['workclass'][j]]  for j in categorical_subset_allfeatures[categorical_subset_allfeatures['income']==i].index]
        d_all[i] = np.asarray(d_all[i], dtype=np.int32)
    
    svm1_all.fit(data=categorical_subset_allfeatures_train)
    
    features_test_split_all = categorical_subset_allfeatures_test.drop('income')
    
    features_predict_test_all = svm1.predict(features_test_split_all)
    count_predict_all = 0
    for p in features_predict_test_all:
        if features_predict_test_all[p] == categorical_subset_allfeatures_test['income'][p]:
            count_predict_all = count_predict_all + 1
    accuracy_all = count_predict_all / len(features_predict_test_all)

total_accuracy_all = accuracy_all/10    
svm1.visualize(data = d)

