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


def make_one_hot(df):
  df['Code type local'] = df['Code type local'].fillna(0)
  df['Type local'] =df['Type local'].fillna('none')
  encode = OneHotEncoder(handle_unknown='ignore').fit(df[['Type local']])
  encoded = encode.transform(df[['Type local']])
  df_encoded = pd.DataFrame(encoded.toarray(), columns = ['is_'+i for i in encode.categories_[0]])
  df_dupli_1hot = pd.concat([df, df_encoded], axis=1)
  return df_dupli_1hot

def make_dummies(df):
  df['Code type local'] = df['Code type local'].fillna(0)
  df['Type local'] =df['Type local'].fillna('none')
  dummies_local = pd.get_dummies(df['Type local'], prefix='nb')
  df_out = pd.concat([df, dummies_local], axis=1)
  return df_out


def m2_terrain(df:pd.DataFrame):
  df.insert(loc=df.shape[1]-1,column='m2_terrain',value=0)
  df.loc[(df['Surface terrain']!=df['Surface terrain'].shift(-1),'m2_terrain')]=df['Surface terrain']
  return df

def modif_colname(df):
  df.columns = map(lambda x:x.lower().replace(' ','_'), df.columns)
  return df

def make_m2(df:pd.DataFrame):
  df.insert(loc=df.shape[1]-1,column='m2_appartement',value=0)
  df['m2_appartement']=df['surface_bati']*df['nb_appartement']
  df.insert(loc=df.shape[1]-1,column='m2_maison',value=0)
  df['m2_maison']=df['surface_bati']*df['nb_maison']
  df.insert(loc=df.shape[1]-1,column='m2_local',value=0)
  df['m2_local']=df['surface_bati']*df['nb_local']
  return df


def effectif_moyen(cluster_model):
  return len(cluster_model.labels_)/len(set(cluster_model.labels_))


def mad(cluster_model):
  unique, counts = np.unique(cluster_model.labels_, return_counts=True)
  mad_score = 0
  for i in counts:
    for j in counts:
      mad_score = mad_score + abs(i-j)
  mad_score = mad_score/len(counts)**2
  return mad_score


def gini_cluster(cluster_model):
  gini = mad(cluster_model)/(2*effectif_moyen(cluster_model))
  return gini
