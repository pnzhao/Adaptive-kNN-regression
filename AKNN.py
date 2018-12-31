# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 09:34:54 2018

@author: puning
"""

import numpy as np
from sklearn.neighbors import KDTree

class AKNNRegressor:
    def __init__(self,K,A,q):
        self.K=K
        self.A=A
        self.q=q
    def fit(self,X,y):
        tree=KDTree(X)
        self.tree=tree
        self.Xtrain=X
        self.ytrain=y
    def predict(self,X):
        tree=self.tree
        ytrain=self.ytrain
        K=self.K
        A=self.A
        q=self.q
        N=len(X)
        ypredict=[0]*N
        nlist=tree.query_radius(X,r=A,count_only=True)
        for i in range(N):
            ka=np.floor(K*(nlist[i]**q)).astype(int)+1
            dist,ind=tree.query(X[i].reshape(-1,1),k=ka)
            ypredict[i]=np.mean(ytrain[ind])
        return np.array(ypredict).reshape(-1,1)