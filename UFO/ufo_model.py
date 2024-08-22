import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read the data

data=pd.read_csv('ufos.csv')
print(data.head(5))

print(data.columns)

# create a small dataframe foronly the columns we need
new_data=pd.DataFrame({'Seconds':data['duration (seconds)'],
                   'Country':data['country'],'Latitude':data['latitude'],
                   'Longitude':data['longitude']})

print(new_data.head())


print(new_data.Country.unique())


new_data.dropna(inplace=True)

print(new_data.info())

print(new_data.Country.unique())


data1=new_data[(new_data['Seconds']>=1) &(new_data['Seconds'] <=60)]
print(data1.head(3))

#Import Scikit-learn's `LabelEncoder` library to convert the text values for countries to a number:
from sklearn.preprocessing import LabelEncoder

data1['Country'] =LabelEncoder().fit_transform(data1['Country'])

print(data1.head(3))

X=data1.drop('Country',axis='columns')
print(X.head(3))

y=data1.Country

print(y.head(3))

print(X.info())
print(y.info())

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.33)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))


print(model.predict([[20,28.9,-96.6]]))
# output was 4 which is country code forthe US

import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))