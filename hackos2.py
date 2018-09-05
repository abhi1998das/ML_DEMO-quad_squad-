import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC,NuSVC
from sklearn import tree

import json

df=pd.read_csv('clubrec.csv')
#print(df)
df.drop(columns =['name'],inplace=True)
x = np.array(df.drop(columns =['drama_club','dance_club','code_club','photography_club','Science_club']))
y1 = np.array(df['drama_club'])
y2 = np.array(df['dance_club'])
y3 = np.array(df['code_club'])
y4 = np.array(df['photography_club'])
y5 = np.array(df['Science_club'])
x_train,x_test,y1_train,y1_test = cross_validation.train_test_split(x,y1,test_size=0.1)
x_train,x_test,y2_train,y2_test = cross_validation.train_test_split(x,y2,test_size=0.1)
x_train,x_test,y3_train,y3_test = cross_validation.train_test_split(x,y3,test_size=0.1)
x_train,x_test,y4_train,y4_test = cross_validation.train_test_split(x,y4,test_size=0.1)
x_train,x_test,y5_train,y5_test = cross_validation.train_test_split(x,y5,test_size=0.1)
##f=0
##clf = MultinomialNB()
##clf.fit(x_train,y1_train)
##f+=clf.score(x_test,y1_test)
##clf.fit(x_train,y2_train)
##f+=clf.score(x_test,y2_test)
##clf.fit(x_train,y3_train)
##f+=clf.score(x_test,y3_test)
##clf.fit(x_train,y4_train)
##f+=clf.score(x_test,y4_test)
##clf.fit(x_train,y5_train)
##f+=clf.score(x_test,y5_test)
##print(f/5*100)
##
##
##f=0
##clf = LogisticRegression()
##clf.fit(x_train,y1_train)
##f+=clf.score(x_test,y1_test)
##clf.fit(x_train,y2_train)
##f+=clf.score(x_test,y2_test)
##clf.fit(x_train,y3_train)
##f+=clf.score(x_test,y3_test)
##clf.fit(x_train,y4_train)
##f+=clf.score(x_test,y4_test)
##clf.fit(x_train,y5_train)
##f+=clf.score(x_test,y5_test)
##print(f/5*100)
##
##
##f=0
##clf = SGDClassifier()
##clf.fit(x_train,y1_train)
##f+=clf.score(x_test,y1_test)
##clf.fit(x_train,y2_train)
##f+=clf.score(x_test,y2_test)
##clf.fit(x_train,y3_train)
##f+=clf.score(x_test,y3_test)
##clf.fit(x_train,y4_train)
##f+=clf.score(x_test,y4_test)
##clf.fit(x_train,y5_train)
##f+=clf.score(x_test,y5_test)
##print(f/5*100)
##
##
f=0
clf = SVC()
clf.fit(x_train,y1_train)
f+=clf.score(x_test,y1_test)
clf.fit(x_train,y2_train)
f+=clf.score(x_test,y2_test)
clf.fit(x_train,y3_train)
f+=clf.score(x_test,y3_test)
clf.fit(x_train,y4_train)
f+=clf.score(x_test,y4_test)
clf.fit(x_train,y5_train)
f+=clf.score(x_test,y5_test)
print(f/5*100)


##f=0
##clf = NuSVC()
##clf.fit(x_train,y1_train)
##f+=clf.score(x_test,y1_test)
##clf.fit(x_train,y2_train)
##f+=clf.score(x_test,y2_test)
##clf.fit(x_train,y3_train)
##f+=clf.score(x_test,y3_test)
##clf.fit(x_train,y4_train)
##f+=clf.score(x_test,y4_test)
##clf.fit(x_train,y5_train)
##f+=clf.score(x_test,y5_test)
##print(f/5*100)


f=0
clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y1_train)
f+=clf.score(x_test,y1_test)
clf.fit(x_train,y2_train)
f+=clf.score(x_test,y2_test)
clf.fit(x_train,y3_train)
f+=clf.score(x_test,y3_test)
clf.fit(x_train,y4_train)
f+=clf.score(x_test,y4_test)
clf.fit(x_train,y5_train)
f+=clf.score(x_test,y5_test)
print(f/5*100)





