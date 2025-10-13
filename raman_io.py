#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from tqdm import tqdm
from scipy.io import loadmat
import pandas as pd
import numpy as np
from scipy.sparse import diags
from scipy import signal
from sklearn.preprocessing import MinMaxScaler,StandardScaler



def airPLS(df, lambda_ = 100000, porder = 2, itermax = 20, wep = 0.1, p = 0.05):
    X=np.array(df)
    m=X.shape[0]
    n=X.shape[1]
    wi1=np.arange(0,np.ceil(wep*n))
    wi2=np.arange(np.floor(n-n*wep)-1,n)
    wi=list(np.concatenate((wi1,wi2), axis=0).astype('int'))
    E=np.eye(n,n)
    E=np.diff(E,n=porder)
    DD=lambda_*np.dot(E,E.T)
    Z=np.zeros((m,n))
    for i in tqdm(range(m)):
        w=np.ones(n)
        x=X[i,:]##
        for j in range(1,itermax+1):
            W = diags(w ,offsets = 0,shape = (n,n))
            C=np.linalg.cholesky(W+DD).T
            z=np.linalg.solve(C,np.linalg.solve(C.T,w*x))
            d=x-z
            dssn=np.abs(d[d<0].sum())
            if(dssn<(0.001*np.abs(x).sum())):
                break
            w[d>=0]=0
            w[wi]=p
            w[d<0] = j*np.exp(abs(d[d<0])/dssn)
        Z[i,:]=z
    Xc = X - Z
    return Xc, Z

def SG(x,width=25,polynomial=2):  
    #Multidimensional data smoothing function based on Savitzky-Golay (SG) filter
    sg=[]
    ar = np.array(x)
    m = x.shape[0]
    for i in range(m):
        ar_i=ar[i, :]
        SG_=signal.savgol_filter(ar[i,:], width, polynomial)
        sg.append(SG_)
    return np.array(sg)

def vector_normalization(data):
    if isinstance(data, pd.DataFrame):
        data = data.values
    x_mean = np.mean(data, axis=1)  
    x_means = np.tile(x_mean, data.shape[1]).reshape((data.shape[0], data.shape[1]), order='F')
    x_2 = np.power(data, 2)
    x_sum = np.sum(x_2, axis=1)
    x_sqrt = np.sqrt(x_sum)
    x_low = np.tile(x_sqrt, data.shape[1]).reshape((data.shape[0], data.shape[1]), order='F')
    return (data - x_means) / x_low

def MMS(x):
    return MinMaxScaler().fit_transform(x.T).T

def SS(x):
    return StandardScaler().fit_transform(x)

class RamanDataset():
    def __init__(self, input_csv, output_csv):
        self.input_csv = input_csv
        self.output_csv = output_csv

    def read_data(self, mat_key=None):
        ext = os.path.splitext(self.input_csv)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(self.input_csv)
        elif ext == '.tsv':
            df = pd.read_csv(self.input_csv, sep='\t')
        elif ext == '.mat':
            mat = loadmat(self.input_csv)
            if not mat_key:
                raise ValueError("The key name (mat_key) of the spectral data contained in the MAT file must be specified.")
            data = mat[mat_key]
            df = pd.DataFrame(data)
        else:
            raise ValueError("The supported file formats are only.csv,.tsv, and.mat")
        return df

    def progre_data(self, mn_mehtod="SS", mat_key=None, fit=False, width=7,polynomial=2):
        #By default, the first column is "ID", the second to the second-to-last columns are spectral data, and the last column is "label"
        df = self.read_data(mat_key=mat_key)
        spectra = df.iloc[:, 1:-1].values
        x = spectra.astype(np.float64)
        labels = df.iloc[:, -1]
        
        Xc,Z=airPLS(x,lambda_ = 100000, porder = 2)
        
        Xsg = SG(Xc,width=width,polynomial=polynomial)
        
        self.mn_mehtod = mn_mehtod
        if self.mn_mehtod=="MMS":
            if fit:
                scaler = MinMaxScaler()
            Xnm = scaler.fit_transform(Xsg)
        elif mn_mehtod=="VNS":
            Xnm= vector_normalization(Xsg)
        elif mn_mehtod=="SS":
            if fit:
                scaler = StandardScaler()
            Xnm = scaler.fit_transform(Xsg)           
        else:
            raise ValueError("Unsupported normalization methods, only support MMS(MinMaxScaler), VNS(vector_normalization), SS(StandardScaler)")
        
        df.iloc[:,1:-1]=Xnm
        df.to_csv(self.output_csv, index=False)
        print(f"✅ The data processing has been completed and saved to:{self.output_csv}")
        
