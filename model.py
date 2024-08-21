import pandas as pd
import numpy as np
import pickle

# read the data
data=pd.read_csv("ml_data.csv")
print(data.head(5))

X=data.drop(['area_type','location','price'],axis="columns")
print(X.head())

y=data.price
print(y.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.33)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


pipeline=make_pipeline(PolynomialFeatures(2),LinearRegression())

pipeline.fit(X_train,y_train)

predict=pipeline.predict(X_test)

poly_score=pipeline.score(X_train,y_train)

print('model_score:',poly_score)


def predict_price(location, total_sqft, bath, bhk):
    global X  # Use your actual DataFrame
    loc_index = X.columns.get_loc(location)

    # Initialize feature values
    X = pd.DataFrame(0, index=[0], columns=X.columns)
    X['total_sqft'] = total_sqft
    X['bath'] = bath
    X['bhk'] = bhk

    if loc_index >= 0:
        X[location] = 1

    return pipeline.predict(X)[0]


print(predict_price('cat__location_Whitefield',1000,2,2))


print(predict_price("cat__location_Old Airport Road",910,4,4))

print(predict_price("cat__location_Old Airport Road",910,3,3))

print(predict_price("cat__location_Lingadheeranahalli",1521.000000,3.0,3.0))


 #Make pickle file of our model
#pickle.dump(pipeline, open("house_model.pkl", "wb"))

import pickle
model_filename = 'house-model.pkl'
pickle.dump(pipeline, open(model_filename,'wb'))

model = pickle.load(open('house-model.pkl','rb'))