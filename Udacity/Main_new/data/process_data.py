import sys
import pandas as pd
from sqlalchemy import create_engine, text



def load_data(messages_filepath, categories_filepath):
    '''input-> messages filepath and categories file path
    
    returns:> merged dataframe on given ID
    
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv( categories_filepath)
    # merge datasets
    df = messages.merge(categories,on='id')
    #Returning merged dataFrame
    return df
   


def clean_data(df):
    ''' This Function takes merged dataFrame as a input
    and return dataFrame with 36 additional columns'''
    column_df=df['categories'].apply(lambda x: x.split(';'))
#new_column_df=column_df
    dic={}
    for row in column_df:
        for value in row:
            col_name=value.split('-')
            if col_name[0] in dic:
                dic[col_name[0]].append(col_name[1]) 
            else:
                dic[col_name[0]]=[]
    #Converting Dictionary to dataframe   
    new_col_df=pd.DataFrame.from_dict(dic)
    df=pd.concat([df.drop(['categories'],axis =1),new_col_df],axis=1)
    df.drop_duplicates(keep=False,inplace=True)
    df['related']=df['related'].replace('2','1')
    
    return df


def save_data(df, database_filename):
    
    ''''
    input-> DataFrame and Database file path
    Return-> Database file path with clean datasets
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, index=False,if_exists='replace')
    return database_filename


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()