import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import * from emal
import json

df = pd.read_csv('studata.csv')
df.replace('IT',1,inplace=True)
df.replace('CSE',2,inplace=True)
df.replace('ECE',3,inplace=True)
df.replace('Printing',4,inplace=True)
df.replace('Chemical',5,inplace=True)
df.replace('Power',6,inplace=True)
df.drop(columns =['NAME','SEM_MARKS'],inplace=True)
x=np.array(df.drop(columns =['Grade']))
y = np.array(df['Grade'])
x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.05)


df2 = pd.read_csv('pydata1.csv')
x2=np.array(df2)
x3=np.array(df2.drop(columns =['Grade','NAME','SEM_MARKS']))
y3 = np.array(df['Grade'])
clf10 = tree.DecisionTreeClassifier()
clf10.fit(x_train,y_train)

clf=MLPClassifier()
clf.fit(x_train,y_train)

lst=[]
pre=clf10.predict(x3)
a=len(x3)
cnt1=0
lst2=[]
for i in range(a):
    if pre[i]>'E':
        lst.append(x2[i][0])
        #send_mail()
        lst2.append(pre[i])

json_string = json.dumps(lst)
json_string2 = json.dumps(lst2)


