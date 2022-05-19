import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


import numpy as np
class LongitudinalDataFrame():

    def __init__(self,source,target,id,time):
        """
        ....
        """
        self.df = source
        self.target = target
        self.id = id
        self.time = time

    def time_shift_dataframe(self,drop_last_value=True):
        df_new = self.df.copy(deep=True)
        df_new.loc[:,'target'] = self.df.groupby(self.id)[self.target].shift(-1)
        df_new.loc[:,'t1'] = self.df.groupby(self.id)[self.time].shift(-1)
        if drop_last_value:
            df_new = df_new.groupby('id', as_index=False).apply(lambda x: x.iloc[:-1])
        df_new.reset_index(drop=True,inplace=True)
        df_new.rename(columns={self.time:'t0'},inplace=True)
        first_cols = [self.id,'t0','t1','target',self.target,]
        self.df = df_new.loc[:,first_cols + list(df_new.columns.difference(first_cols))]

    def split(self,to_numpy=True,to_torch=False):
        X = self.df.loc[:,self.target]
        y = self.df.target
        dt = self.df.loc[:,['t0','t1']]
        if to_numpy or to_torch:
            X = np.array(X).astype(np.float32)
            y = np.array(y).astype(np.float32)
            dt = np.array(dt).astype(np.float32)
        if to_torch:
            X = torch.Tensor(X)
            y = torch.Tensor(y)
            dt = torch.Tensor(dt)
        return X,y,dt

class LongitudinalDataset(Dataset):

    def __init__(self,df,target,id,time):
        """
        Expects data in form...
        """
        self.df = df
        self.target = target
        self.id = id 
        self.time = time

        df = self._create_target_dataframe(df,target,id,time)

        self.X,self.y,self.dt = self.load_data(df)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # pad
        X = self.X[idx].astype(np.float32)
        y = self.y[idx].astype(np.float32)
        dt = self.dt[idx].astype(np.float32)

        return X,y,dt

    def _create_target_dataframe(self,df,target,id,time,drop_last_value=True):
        df = df.copy()
        df['target'] = df.groupby(id)[target].shift(-1)
        df['t1'] = df.groupby(id)[time].shift(-1)
        if drop_last_value:
            df = df.groupby('id', as_index=False).apply(lambda x: x.iloc[:-1])
        df.reset_index(drop=True,inplace=True)
        df.rename(columns={time:'t0'},inplace=True)
        first_cols = [id,'t0','t1','target',target,]
        df = df.loc[:,first_cols + list(df.columns.difference(first_cols))]
        return df

    def load_data(self,df):

        X_list,y_list,dt_list = [],[],[]
        ids = df.id.unique()
        for id_ in ids:
            # extract from dataframe
            df_id = df.loc[df.id == id_,:]
            X = df_id.loc[:,[self.target]]
            y = df_id.loc[:,'target']
            dt = df_id.loc[:,['t0','t1']]
            # to numpy
            X = np.array(X).astype(np.float32)
            y = np.array(y).astype(np.float32)
            dt = np.array(dt).astype(np.float32)
            # append to lists
            X_list.append(X)
            y_list.append(y)
            dt_list.append(dt)
        return X_list,y_list,dt_list

def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''
    ## get sequence lengths
    lengths = torch.tensor([b[0].shape[0] for b in batch])
    ## padding
    x = [torch.Tensor(b[0]) for b in batch]
    y = [torch.Tensor(b[1]) for b in batch]
    dt = [torch.Tensor(b[2]) for b in batch]
    x = torch.nn.utils.rnn.pad_sequence(x,batch_first=True)
    y = torch.nn.utils.rnn.pad_sequence(y,batch_first=True)
    dt = torch.nn.utils.rnn.pad_sequence(dt,batch_first=True)
    return x,y,dt
