from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
import time



def PD(state):
    # return a array of probability distribution of given state
    
    return np.transpose( np.power(np.absolute( np.array(state) ),2) )[0]

def initialstate(K):
    # return a tensor of 2*K spin down states
    
    statelist = []
    for i in range(2*K):
        statelist.append(basis(2,0))
    
    return tensor( statelist )

def plotPD( state ):
    # plot the probability distribution of the given state
    
    dim = np.shape(state)[0]  # dimension of the state
    plt.plot( range(dim), PD(state) )

def plotPDsorted( state ):
    # plot the "sorted" probability distribution of the given state
    
    dim = np.shape(state)[0]  # dimension of the state
    plt.plot( range(dim), np.sort( PD(state) ) )
    
def plotPDlog( state ):
    # plot the log10 of the "sorted" probability distribution of the given state
    
    dim = np.shape(state)[0]  # dimension of the state
    plt.plot( range(dim), np.log10( np.sort( PD(state) ) ) )

def XXoper(theta, K):    
    # applying sigmax()sigmax() to every adjacent qubits
    
    XXlist1 = []
    for i in range(K):
        XXlist1.append( np.cos(theta) * tensor( identity(2) , identity(2) ) + 1j * np.sin(theta) * tensor( sigmax() , sigmax() ) )
    
    XXlist2 = [ identity(2) ]
    for j in range(K-1):
        XXlist2.append( np.cos(theta) * tensor( identity(2) , identity(2) ) + 1j * np.sin(theta) * tensor( sigmax() , sigmax() ) )
    XXlist2.append( identity(2) )
    
    return tensor(XXlist1) * tensor(XXlist2)

def Zoper(theta, K):
    # applying sigmaz() to every qubit
    
    Zlist = []
    for i in range(2*K):
        Zlist.append( np.cos(theta) * identity(2) + 1j * np.sin(theta) * sigmaz() )
        
    return tensor(Zlist)

def Xoper(theta, K):
    # applying sigmax() to every qubit
    
    Xlist = []
    for i in range(2*K):
        Xlist.append( np.cos(theta) * identity(2) + 1j * np.sin(theta) * sigmax() )
        
    return tensor(Xlist)

def entropy(state):
    Parray = PD(state)  + np.power(0.1,12)
    logParray = -np.log(Parray)
    return np.sum( np.dot(Parray, logParray) )

def GaussianTheta(num_seq, sigma, num_instance):
    # output is a matrix of num_instance times num_seq, each row is one instance of Theta(sequence)
    
    np.random.seed()
    
    Delta = scipy.linalg.toeplitz( - np.arange(num_seq), np.arange(num_seq) )
    # 用index距离作为矩阵元的 toeplitz 矩阵
    
    CovMat = np.exp( - np.power(Delta,2) / ( 2 * sigma ** 2 ) )
    # 将上述矩阵的矩阵元作为高斯函数的自变量，成为 Covariance matrix
    
    eig_val, eig_vec = np.linalg.eigh(CovMat) # 求得其 本征值，本征矢量
    eig_val = np.maximum(eig_val, 0.0) # 保证 本征值不为负数
    
    L = np.dot( np.sqrt(np.diag(eig_val)) , eig_vec.T )
    K = np.dot( np.random.randn( num_instance, num_seq ) , L )
    
    return K