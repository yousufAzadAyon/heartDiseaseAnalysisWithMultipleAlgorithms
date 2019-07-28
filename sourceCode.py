#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.svm import SVC
#%%
df=pd.read_csv('heart.csv')
df.head(5)

#%%
df.describe()

#%%
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,fmt='.1f')
plt.show()
#%%
df.target.value_counts()

#%%
sns.countplot(x="target", data=df, palette="bwr")
plt.show()

#%%
sns.distplot(df['age'],rug=True)
plt.show()


#%%
df.sex.value_counts()

#%%
sns.countplot(x='sex', data=df)
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()

#%%
plt.figure(num=None, figsize=(7, 4))

sns.barplot(y='thalach', x='target',hue='sex', data=df)
plt.show()

#%%
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(10,5),color=['#1CA53B','#AA1111' ])
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


#%%
sns.countplot(x='cp', data=df)
plt.xlabel('Chest Pain Type')
plt.show()

#%%
pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190' ])
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()

#%%
sns.barplot(x=df.thalach.value_counts()[:10].index,y=df.thalach.value_counts()[:10].values)
plt.xlabel('max heart rate')
plt.ylabel('Counter')
plt.show()

#%%
chestPain=pd.get_dummies(df['cp'],prefix='cp',drop_first=True)
df=pd.concat([df,chestPain],axis=1)
df.drop(['cp'],axis=1,inplace=True)
sp=pd.get_dummies(df['slope'],prefix='slope')
th=pd.get_dummies(df['thal'],prefix='thal')
frames=[df,sp,th]
df=pd.concat(frames,axis=1)
df.drop(['slope','thal'],axis=1,inplace=True)

#%%
df.head(5)

#%%
X = df.drop(['target'], axis = 1)
y = df.target.values

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#%%
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#%%
#LogisticRegression
logiReg=LogisticRegression(random_state=0)
logiReg.fit(X_train,y_train)
logiRegPred=logiReg.predict(X_test)
logiRegAcu=accuracy_score(y_test, logiRegPred)

#SVM classifier
svc=SVC(kernel='linear',random_state=0)
svc.fit(X_train,y_train)
svcPred=svc.predict(X_test)
svcAcu=accuracy_score(y_test, svcPred)

#Bayes
bayes=GaussianNB()
bayes.fit(X_train,y_train)
bayesPred=bayes.predict(X_test)
bayesAcu=accuracy_score(bayesPred,y_test)

#SVM regressor
svcReg=SVC(kernel='rbf')
svcReg.fit(X_train,y_train)
svcRedPred=svcReg.predict(X_test)
svcRedAcu=accuracy_score(y_test, svcRedPred)

#RandomForest
randomForest=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
randomForest.fit(X_train,y_train)
randomForestPred=randomForest.predict(X_test)
randomForestAcu=accuracy_score(randomForestPred,y_test)

# DecisionTree Classifier
dTree=DecisionTreeClassifier(criterion='entropy',random_state=0)
dTree.fit(X_train,y_train)
dTreePred=dTree.predict(X_test)
dTreeAcu=accuracy_score(dTreePred,y_test)

#KNN
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
knnPred=knn.predict(X_test)
knnAcu=accuracy_score(knnPred,y_test)

#%%

#%%
print('LogisticRegression_accuracy:\t',logiRegAcu)
print('SVM_regressor_accuracy:\t\t',svcRedAcu)
print('RandomForest_accuracy:\t\t',randomForestAcu)
print('DecisionTree_accuracy:\t\t',dTreeAcu)
print('KNN_accuracy:\t\t\t',knnAcu)
print('SVM_classifier_accuracy:\t',svcAcu)
print('Bayes_accuracy:\t\t\t',bayesAcu)

#%%
model_accuracy = pd.Series(data=[logiRegAcu,svcAcu,bayesAcu,svcRedAcu,randomForestAcu,dTreeAcu,knnAcu], 
index=['LogisticRegression','SVM_classifier','Bayes','SVM_regressor',
'RandomForest','DecisionTree_Classifier','KNN'])
fig= plt.figure(figsize=(10,7))
model_accuracy.sort_values().plot.barh()
plt.title('Model Accracy')

#%%

