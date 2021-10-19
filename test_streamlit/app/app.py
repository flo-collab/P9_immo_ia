import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

arrondissements = [ 1,  2,  3,  4, 10]

filename ='app/reglin_paris_model.sav'
model_regLin = pickle.load(open(filename, 'rb'))

# Streamlit part:

#st.set_page_config(layout="wide")

st.title('Prédictiion de prix immobilier à Paris')

c1,c2,c3 = st.columns(3)


with c1:
    arrondissement = st.selectbox('Arrondissement :',arrondissements)


with c2:
    surface = st.number_input('Surface',min_value=0)
    st.write('Surface du Logement choisie :', surface)

data_for_pred = np.matrix([surface,arrondissement])
my_pred = model_regLin.predict(data_for_pred)

with c3:
    st.header('Predicftion du prix : ')
    st.write(my_pred[0])

