import pickle
from ast import literal_eval
import collections

def load_data(location):
    with open(location, 'rb') as file:
        data = pickle.load(file)
        return data

def save_data(data, location):
    with open(location, 'wb') as file:
        pickle.dump(data,file)

def literal(df,col_names):
    for col in col_names:
        df[col] = df[col].apply(literal_eval)
    return df

def literal_all_cols(df):
    for col in df.columns:
        try:
            if col != 'Subtitle':
                df[col] = df[col].apply(literal_eval)
        except Exception as e:
            pass
    return df

