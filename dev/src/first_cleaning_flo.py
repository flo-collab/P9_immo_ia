import pandas as pd
import numpy as np

def dirty_cleaning(df_original):
    df_immo_full = drop_empty_col(df_original,0.99)
    df_immo_full.dropna(subset=['Valeur fonciere'],inplace=True)
    df_immo_full['Nombre pieces principales'] = df_immo_full['Nombre pieces principales'].fillna(0)
    df_immo_full['Nombre pieces principales'] = df_immo_full['Nombre pieces principales'].astype(int)
    df_immo_full = drop_empty_col(df_original,0.92)
    df_immo_full['Date mutation'] = pd.to_datetime(df_immo_full['Date mutation'])
    df_immo_full['No voie'] = df_immo_full['No voie'].astype('Int64')
    df_immo_full['Nombre pieces principales'] = df_immo_full['Nombre pieces principales'].astype('Int64')
    df_immo_full = df_immo_full.sort_values(['Commune', 'Code postal'], ascending=[True, False])
    df_immo_full['Code postal'] = df_immo_full['Code postal'].fillna(method='ffill')
    df_immo_full['Code postal'] = df_immo_full['Code postal'].astype('Int64')
    df_immo_full['Code postal'] = df_immo_full['Code postal'].astype(str)
    df_immo_full['Valeur fonciere'] = df_immo_full['Valeur fonciere'].str.replace(",",".").astype(float)

    exclude = [0 ,np.nan]
    test_pb = df_immo_full[df_immo_full['Surface terrain'].isin(exclude) & df_immo_full['Surface reelle bati'].isin(exclude)]
    other_pb = df_immo_full[df_immo_full['Surface reelle bati'].isna() & df_immo_full['Type local'].notna()]
    print(df_immo_full.shape)
    print(test_pb.shape)
    df_immo_full.drop(test_pb.index, inplace=True)
    other_pb = df_immo_full[df_immo_full['Surface reelle bati'].isna() & df_immo_full['Type local'].notna()]
    print(other_pb.shape)
    df_immo_full.drop(other_pb.index, inplace=True)
    my_zscore = zscore(df_immo_full['Valeur fonciere'])
    df_immo_full.drop(my_zscore[(abs(my_zscore)>2)].index, inplace=True)
    print(df_immo_full.shape)
    df_immo_full.sort_index(inplace=True)
    df_immo_full = df_immo_full.reset_index().drop(columns = 'index')

    return df_immo_full

def make_one_hot(df_immo_full):
    #OneHot
    df_immo_full['Code type local'] = df_immo_full['Code type local'].fillna(0)
    df_immo_full['Type local'] =df_immo_full['Type local'].fillna('none')
    df_immo_full_cat = df_immo_full[['Type local']]
    from sklearn.preprocessing import OneHotEncoder
    cat_local = OneHotEncoder(handle_unknown='ignore')
    #cat_local.fit(df_immo_full['Type local'].values.reshape(-1, 1))
    df_immo_1hot = cat_local.fit_transform(df_immo_full_cat)
    df_1hot = pd.DataFrame(df_immo_1hot.toarray(), columns = ['is_'+i for i in cat_local.categories_[0]])
    df_immo_1hot = pd.concat([df_immo_full, df_1hot], axis=1)
    df_immo_1hot.drop(columns = 'is_Local industriel. commercial ou assimilé', inplace = True)
    return df_immo_1hot