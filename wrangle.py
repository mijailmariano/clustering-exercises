# importing main libraries/modules
import os
import pandas as pd
import numpy as np

# importing data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()

# importing mysql
import env
from env import user, password, host, get_connection



'''function that will either 
1. import the zillow dataset from MySQL or 
2. import from cached .csv file'''
def get_zillow_dataset():
    # importing "cached" dataset
    filename = "zillow.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=[0])

    # if not in local folder, let's import from MySQL and create a .csv file
    else:
        # query used to pull the 2017 properties table from MySQL
        query = ''' 
                SELECT *
                FROM properties_2017
                        RIGHT JOIN (SELECT parcelid, any_value(logerror), MAX(transactiondate) AS maxtransaction_date 
                            FROM predictions_2017
                                GROUP BY (predictions_2017.parcelid)) AS table2 USING (parcelid)
                                    WHERE maxtransaction_date < 2018'''
        
        db_url = f'mysql+pymysql://{user}:{password}@{host}/zillow'

        # creating the zillow dataframe using Pandas' read_sql() function
        df = pd.read_sql(query, db_url)
        df.to_csv(filename)

        return df



def remove_columns(df, cols_to_remove):
    df = df.drop(columns=cols_to_remove)
    return df


def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.75):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold) #1, or ‘columns’ : Drop columns which contain missing value
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold) #0, or ‘index’ : Drop rows which contain missing values.
    return df


# combining both functions above in a cleaning function:
def data_prep(df, cols_to_remove=[], prop_required_column=0.5, prop_required_row=0.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df