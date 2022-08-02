# importing main libraries/modules
import os
import pandas as pd
import numpy as np

# importing data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()

# sklearn library for data science
from sklearn.model_selection import train_test_split

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
        SELECT prop.*,
        predictions_2017.logerror,
        predictions_2017.transactiondate,
        air.airconditioningdesc,
        arch.architecturalstyledesc,
        build.buildingclassdesc,
        heat.heatingorsystemdesc,
        land.propertylandusedesc,
        story.storydesc,
        type.typeconstructiondesc
        FROM properties_2017 prop
        JOIN (
            SELECT parcelid, MAX(transactiondate) AS max_transactiondate
            FROM predictions_2017
            GROUP BY parcelid
            ) pred USING(parcelid)
        JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                          AND pred.max_transactiondate = predictions_2017.transactiondate
        LEFT JOIN airconditioningtype air USING(airconditioningtypeid)
        LEFT JOIN architecturalstyletype arch USING(architecturalstyletypeid)
        LEFT JOIN buildingclasstype build USING(buildingclasstypeid)
        LEFT JOIN heatingorsystemtype heat USING(heatingorsystemtypeid)
        LEFT JOIN propertylandusetype land USING(propertylandusetypeid)
        LEFT JOIN storytype story USING(storytypeid)
        LEFT JOIN typeconstructiontype type USING(typeconstructiontypeid)
        WHERE propertylandusedesc = "Single Family Residential"
            AND transactiondate <= '2017-12-31'
            AND prop.longitude IS NOT NULL
            AND prop.latitude IS NOT NULL
        '''
        
        db_url = f'mysql+pymysql://{user}:{password}@{host}/zillow'

        # creating the zillow dataframe using Pandas' read_sql() function
        df = pd.read_sql(query, db_url)
        df.to_csv(filename)

        return df


'''Function takes in a dataframe and returns a feature/column total null count and percentage df'''
def null_df(df):
    # creating a container to hold all features and needed null data
    container = []

    for col in list(df.columns):
        feature_info = {
            "Name": col, \
            "Total Null": df[col].isnull().sum(), \
            "Feature Null %": df[col].isnull().sum() / df.shape[0]
        }
        # appending feature and data to container list
        container.append(feature_info)
        
    # creating the new dataframe
    new_df = pd.DataFrame(container)

    # setting the df index to "name"
    new_df = new_df.set_index("Name")

    # setting index name to None
    new_df = new_df.rename_axis(None, axis = 0)

    # sorting df by percentage of null descending
    new_df = new_df.sort_values("Total Null", ascending = False)

    # returning the new null dataframe
    return new_df


# function to drop columns/rows based on proportion of nulls across record and feature
def drop_nulls(df, required_column_percentage, required_record_percentage):
    
    feature_null_percentage = 1 - required_column_percentage
    
    for col in list(df.columns):
        
        null_sum = df[col].isna().sum()
        null_pct = null_sum / df.shape[0]
        
        if null_pct > feature_null_percentage:
            df.drop(columns=col, inplace=True)
            
    feature_threshold = int(required_record_percentage * df.shape[1])
    
    df = df.dropna(axis = 0, thresh = feature_threshold)
    
    return df

'''Function created to split the initial dataset into train, validate, and test datsets'''
def train_validate_test_split(df):
    train_and_validate, test = train_test_split(
    df, test_size = 0.2, random_state = 123)
    
    train, validate = train_test_split(
        train_and_validate,
        test_size = 0.3,
        random_state = 123)

    print(f'train shape: {train.shape}')
    print(f'validate shape: {validate.shape}')
    print(f'test shape: {test.shape}')

    return train, validate, test


'''-----------------------------------'''
# borrowed/previous lesson functions

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