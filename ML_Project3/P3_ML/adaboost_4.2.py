import math
import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv("adult.csv")

data = data[(data != '?').all(1)]

x = LabelEncoder()

#transforimng discrete features
workclassparam = x.fit_transform(data['workclass'])
educationparam = x.fit_transform(data['education'])
martialparam = x.fit_transform(data['marital.status'])
occupationparam = x.fit_transform(data['occupation'])
relationshipparam = x.fit_transform(data['relationship'])
raceparam = x.fit_transform(data['race'])
nativeparam = x.fit_transform(data['native.country'])
incomeparam = x.fit_transform(data['income'])
sexparam = x.fit_transform(data['sex'])

# dataframe after dealing with discrete features
transformed_data = data[['age', 'fnlwgt', 'capital.gain', 'capital.loss', 'hours.per.week']].copy()

transformed_data['workclass'] = workclassparam
transformed_data['education'] = educationparam
transformed_data['marital.status'] = martialparam
transformed_data['occupation'] = occupationparam
transformed_data['relationship'] = relationshipparam
transformed_data['race'] = raceparam
transformed_data['native.country'] = nativeparam
transformed_data['income'] = incomeparam
transformed_data['sex'] = sexparam
features = list(set(transformed_data.columns) - set(['income']))
cat_subset = pd.get_dummies(data, columns=['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'native.country'])
cat_subset.drop('education.num', axis =1)
sex_category = x.fit_transform(cat_subset['sex'])
income_category = x.fit_transform(data['income'])

cat_subset.drop('sex', axis =1)
cat_subset.drop('income', axis =1)
cat_subset['sex'] = sex_category
cat_subset['income'] = income_category
X = cat_subset.drop(['income'],axis=1)
Y = cat_subset['income']
AdaBoost = Ada_Boost_Classifier(n_estimators=400,learning_rate=1,algorithm='SAMME')
AdaBoost.fit(X,Y)
prediction = AdaBoost.score(X,Y)

print('The accuracy is: ',prediction*100,'%')
