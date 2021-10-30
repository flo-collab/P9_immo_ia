import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# On charge le dataset
house_data = pd.read_csv("house_data.csv")
# Ici on enleve les outliers
X = house_data[house_data.columns[1:]]
y = house_data['price']
house_data = house_data.dropna()
house_data = house_data.astype(int)
house_data = house_data[house_data['price']<8000]

#print (house_data.info())

X = house_data[house_data.columns[1:]]
y = house_data['price']

#print (X.head())

#  on divise notre jeu de données en 2 parties
# 80%, pour l’apprentissage et les 20% restant pour le test.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# entrainement  du modèle
from sklearn.metrics import mean_squared_error

model_regLin = LinearRegression()
model_regLin.fit(X_train, y_train)



# on regarde les resultats : Les coefficients
print('Coefficients: \n', model_regLin.coef_)

# Evaluation du training set
from sklearn.metrics import r2_score

y_train_predict = model_regLin.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)
print ('Le score r² est :',r2,'\nLe score RMSE est : ',rmse)


# save the model to disk
filename = 'reglin_paris_model.sav'
pickle.dump(model_regLin, open(filename, 'wb'))


g = sns.lmplot(
    data=house_data,
    x="surface", y="price", hue="arrondissement",col="arrondissement",
    height=5)
#plt.show()


 
#st.title('Prédictiion de prix immobilier à Paris')


print('les arrondissements sont : ',house_data['arrondissement'].unique())

# Streamlit part:

#st.set_page_config(layout="wide")

st.title('Prédictiion de prix immobilier à Paris')

c1,c2,c3 = st.columns(3)


with c1:
    arrondissement = st.selectbox('Arrondissement :',house_data['arrondissement'].unique())


with c2:
    surface = st.number_input('Surface',min_value=0)
    st.write('Surface du Logement choisie :', surface)

data_for_pred = np.matrix([surface,arrondissement])
my_pred = model_regLin.predict(data_for_pred)

with c3:
    st.header('Predicftion du prix : ')
    st.write(my_pred[0])
    





