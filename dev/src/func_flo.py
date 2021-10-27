import pandas as pd
import numpy as np
import time

def nan_rate(df:pd.DataFrame):
  return df.isnull().sum()/df.shape[0]

def treshold_na_col(df:pd.DataFrame,treshold:float):
  return [nan_rate(df).index[i] for i in range(nan_rate(df).shape[0]) if nan_rate(df).values[i] > treshold]

def drop_empty_col(df:pd.DataFrame,treshold:float):
  prop_na = nan_rate(df)
  list_index_col = [prop_na.index[i] for i in range(prop_na.shape[0]) if prop_na.values[i] > treshold]
  df.drop(list_index_col, axis = 1,inplace=True)
  return df

def col_to_int(df:pd.DataFrame,col:str):
  df[col] = df[col].astype(int)
  return df

def col_fill_0(df:pd.DataFrame,col:str):
  df[col] = df[col].fillna(0)
  return df

def check_if_can_be_int(df,col):
  for i in df[col].dropna().values:
    if i%1 >0 :
      print(i)
  return

def zscore(df_col):
  zscore = (df_col - df_col.mean()) / df_col.std()
  return zscore
