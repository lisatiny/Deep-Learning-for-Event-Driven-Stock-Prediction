import pandas as pd
import numpy as np
import torch

class Preprocess(object) :

    def __init__(self,lemma_dict=None) :
        self.lemma_dict = lemma_dict

    def __mapping(self,x) :
        try :
            return self.lemma_dict[x]
        except :
            return 0 # UNK

    def get_indexed(self,df, col, count_threshold=0):
        """
        this methode supports input as pandas dataframe.
        df:pandas dataframe
        col:str => column name of dataframe
        lemma_dict:dictionary => this will be udpated as a inplacement
        count_thresholde:int => type in integar type if you want to do not use less frequent word
            it will be binded to UNK
        """

        count_arr = np.unique(df[col], return_counts=True)
        use_corpus = count_arr[0][count_arr[1] > count_threshold]
        X = df[col].values

        print("[numericalize] lemma dict is being updated")
        for word in use_corpus:
            try:
                self.lemma_dict[word]
            except:
                self.lemma_dict[word] = len(self.lemma_dict)
        indexed_arr = map(self.__mapping,X)
        return indexed_arr

    @staticmethod
    def padding(x):
        how = 3 - len(x)
        padded_x = x + ['PAD'] * how
        return padded_x

    def new_indexing_logic(self,df,col,count_threshold=0) :
        tmp = df[col].str.strip().str.split(" ")
        padded_tmp = tmp.apply(self.padding)

        count_arr = np.unique([j for i in padded_tmp for j in i],return_counts=True)
        corpus_ls = count_arr[0][count_arr[1] > count_threshold]

        for corpus in corpus_ls:
            try:
                self.lemma_dict[corpus]
            except:
                self.lemma_dict[corpus] = len(self.lemma_dict)

        indexed_tmp = padded_tmp.apply(lambda x : map(self.__mapping,x))
        train_X = np.vstack(indexed_tmp.values)

        return train_X