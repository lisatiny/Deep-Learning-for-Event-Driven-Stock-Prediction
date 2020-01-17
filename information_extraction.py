### built-in ###
import numpy as np
import pandas as pd

### extract event ###
from openie import StanfordOpenIE

    
def information_extraction(text) : 
    with StanfordOpenIE() as client:
        output = client.annotate(text)
    if isinstance(output,list) : 
        if len(output) != 0 : 
            return output[0]
        else : 
            return np.nan
    return output

def parsing(input_df) : 
    df = input_df.copy()
    tmp = df.PARSED_HEADLINE.str.replace("corporation|corp[.]*",'')
    tmp = tmp.str.replace("[.]|[,]|\(.+\)",'')
    tmp = tmp.str.replace("\d+",'^NUM')
    tmp = tmp.str.replace("[^A-Za-z0-9\s]+",'')
    df['PARSED_HEADLINE'] = tmp
    return df

def wrapper(path,dump=False) : 
    print("[wrapper] PARSED HEADLINE is going to be extracted S/P/O structure")
    df = pd.read_csv(path)
    print("[wrapper] loading is finished")
    df = parsing(df)
    print("[wrapper] parsing is finished")    
    df['OPENIE'] = df.PARSED_HEADLINE.apply(information_extraction)
    df = df[df.OPENIE.notnull()].reset_index(drop=True)
    print("[wrapper] data set-up is finished {}".format(df.shape))
    df['SUBJECT'] = df.OPENIE.apply(lambda dict_ : dict_['subject'])
    df['RELATION'] = df.OPENIE.apply(lambda dict_ : dict_['relation'])
    df['OBJECT'] = df.OPENIE.apply(lambda dict_ : dict_['object'])
    if dump : 
        df.to_csv(path,index=False)
    else : 
        return df