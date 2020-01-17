### built-in ###
import numpy as np
import pandas as pd
# ### nlp pkgs ###
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

def vanila_parsing(path,low_memory=False) : 
    if low_memory :
        final_df = pd.DataFrame()
        df_iter = pd.read_csv(path,chunksize=100000)
        for chunk_df in df_iter : 
            unique_df = chunk_df.drop_duplicates('TIMESTAMP_EDT')
            unique_df['HEADLINE'] = unique_df.HEADLINE.str.lower()
            unique_df = unique_df.drop_duplicates('HEADLINE')

            fact_df = unique_df[(unique_df.FACT_LEVEL == 'fact') & (unique_df.RELEVANCE > 75)].reset_index(drop=True)
            final_df = pd.concat([final_df,fact_df],ignore_index=True)
        return final_df
    
    else : 
        df = pd.read_csv(path)
        unique_df = df.drop_duplicates('TIMESTAMP_EDT')
        unique_df['HEADLINE'] = unique_df.HEADLINE.str.lower()
        unique_df = unique_df.drop_duplicates('HEADLINE')

        fact_df = unique_df[(unique_df.FACT_LEVEL == 'fact') & (unique_df.RELEVANCE > 75)].reset_index(drop=True)
        return fact_df
    
def func(x) : 
    tmp = lemmatizer.lemmatize(x.decode('utf-8'),'v')
    result = lemmatizer.lemmatize(tmp,'n')    
    return result

def headline_lemmatize(df) : 
    tmp = df.HEADLINE.str.split(" ").apply(lambda sent : map(func,sent))
    tmp = tmp.str.join(" ")
    tmp = tmp.str.replace(',',' ')
    tmp = tmp.apply(lambda x : x.encode("utf-8"))
    df['PARSED_HEADLINE'] = tmp      
    return df

def wrapper(path,dump=False) : 
    df = vanila_parsing(path,low_memory=True)
    print("[wrapper] parsing is over")
    df = headline_lemmatize(df)
    print("[wrapper] lemmatizing is over")
    if dump : 
        df.to_csv(path,index=False)
    else : 
        return df