# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 09:35:28 2018

@author: puning
"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from AKNN import AKNNRegressor
import pandas as pd
from tqdm import tqdm
def Generate(N,dist,datatype,bound):  
    if dist=='Laplace':
        x=np.random.laplace(0,1,size=N)
    if dist=='Gaussian':
        x=np.random.normal(0,1,size=N)
    if dist=='Cauchy':
        x=np.random.standard_cauchy(size=N)
    if dist=='t2':
        x=np.random.standard_t(2,N)
    if dist=='t3':
        x=np.random.standard_t(3,N)
    if bound==True:
        y=np.sin(x)
    else:
        y=x
    if datatype=='train':
        noise=np.random.normal(0,1,size=N) # Set noise level.
        y=y+noise
    return [x,y]
def evaluate(model,dtrain,dtest):
    xtrain=dtrain[0].reshape(-1,1)
    ytrain=dtrain[1].reshape(-1,1)
    xtest=dtest[0].reshape(-1,1)
    ytest=dtest[1].reshape(-1,1)
    model.fit(xtrain,ytrain)
    pred=model.predict(xtest)
    err=ytest-pred
    return np.mean(err**2)
def bestk(M,Ntrain,Ntest,dist,bound,kcandidates):
    L=len(kcandidates)
    R=[[0 for j in range(M)] for i in range(L)]
    for j in tqdm(range(M)):
        dtrain=Generate(Ntrain,dist,'train',bound)
        dtest=Generate(Ntrain,dist,'test',bound)
        for i in range(L):
            k=kcandidates[i]
            model=KNeighborsRegressor(n_neighbors=k,weights='uniform')
            R[i][j]=evaluate(model,dtrain,dtest)
    Risk=[np.mean(R[i]) for i in range(L)]
    kopt=kcandidates[np.argmin(Risk)]
    print('For the following k:')
    print(kcandidates)
    print('The estimated risks are:')
    print(Risk)
    print('Selected:',kopt)
    return kopt
def bestK(M,Ntrain,Ntest,dist,bound,Kcandidates):
    L=len(Kcandidates)
    R=[[0 for j in range(M)] for i in range(L)]
    for j in tqdm(range(M)):
        dtrain=Generate(Ntrain,dist,'train',bound)
        dtest=Generate(Ntrain,dist,'test',bound)
        for i in range(L):
            model=AKNNRegressor(K=Kcandidates[i],A=0.5,q=0.8)
            R[i][j]=evaluate(model,dtrain,dtest)
    Risk=[np.mean(R[i]) for i in range(L)]
    Kopt=Kcandidates[np.argmin(Risk)]
    print('For the following k:')
    print(Kcandidates)
    print('The estimated risks are:')
    print(Risk)
    print('Selected:',Kopt)
    return Kopt
def compute(distribution,boundstatus):
    k_list=[5,10,15,20,25,30,50]
    K_list=[0.5,1,1.5,2,2.5,3,5]
    N_list=[500,1000,2000,5000,10000,20000,50000]
    Ntest=1000
    M=500
    L=len(N_list)
    kopt=bestk(M=300,Ntrain=500,Ntest=1000,dist=distribution,bound=boundstatus,kcandidates=k_list)
    Kopt=bestK(M=300,Ntrain=500,Ntest=1000,dist=distribution,bound=boundstatus,Kcandidates=K_list)
    if distribution in ['Laplace','Gaussian']:
        growth_index=0.5
    if distribution=='Cauchy':
        growth_index=1/3
    if distribution=='t2':
        growth_index=0.4
    if distribution=='t3':
        growth_index=3/7
    R=[[0 for l in range(3)] for i in range(L)]
    model2=AKNNRegressor(K=Kopt,A=0.5,q=0.8)
    for i in range(L):
        N=N_list[i]
        ka=round(kopt*(N/500)**growth_index)
        model1=KNeighborsRegressor(n_neighbors=ka,weights='uniform')
        print('Training sample size:',N)
        print('k used in standard Nearest Neighbor:',ka)
        err1=[0]*M
        err2=[0]*M
        for j in tqdm(range(M)):
            dtrain=Generate(N,distribution,'train',boundstatus)
            dtest=Generate(Ntest,distribution,'test',boundstatus)
            err1[j]=evaluate(model1,dtrain,dtest)
            err2[j]=evaluate(model2,dtrain,dtest)
        R[i][0]=N
        R[i][1]=np.mean(err1)
        R[i][2]=np.mean(err2)
        print('Estimated risk of standard kNN:',R[i][1])
        print('Estimated risk of adaptive kNN:',R[i][2])
    print('Final Result:',R)
    Output=pd.DataFrame(R)
    if boundstatus==True:
        filename=distribution+'Bounded.csv'
    else:
        filename=distribution+'Unbounded.csv'
    Output.to_csv(filename,header=['Ntrain','Std','Ada'])
                
compute('t3',False)   
        