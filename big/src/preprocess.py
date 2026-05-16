import pandas as pd

def load_data():
    df = pd.read_csv("data/customers.csv")
    
    # تحويل gender
    df['gender'] = df['gender'].map({'M':0 , 'F':1})
    
    return df