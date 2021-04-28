#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'


import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
import time
import multiprocessing
from qutip import *


from NNGS_funs import *

def mainQ(index,num_pair,num_seq,sigma):
    print("mainQ",index)
    state = initialstate(M) # this is the inital state of product state, you could choose to change it to a initial state of PT distribution
    Prob_Seq = PD(state)

    ThetaX = GaussianTheta( num_seq, sigma, 1)

    for j in range(1, sequence + 1):

        xtheta = ThetaX[0, j - 1]

        state = XXoper(np.pi / 4, num_pair) * state
        state = Zoper(np.pi / 4, num_pair) * state
        state = Xoper(xtheta, num_pair) * state
        
        Prob_Seq = np.block( [ [Prob_Seq] , [PD(state)] ] )

    np.savez( str(2*M) + "qubits" + str(sequence) + "sequcences" + str(correlation) + "sigma_trajectory" + str(index) + ".npz", ThetaX = ThetaX, Seq_Prob = Prob_Seq)
    # each trajectory creates one file, after that regroup.py is used to combine them into one file

#q = multiprocessing.Queue()

M = 5  # number of pairs of qubits
sequence = 40 # number of sequence 
correlation = 20 # correlation length
N = 40000 # number of trajectories

ti = time.time()



if __name__ == "__main__":
    
    processes = [multiprocessing.Process(target=mainQ, args=(x,M,sequence,correlation,)) for x in range(N)]
    #print(processes)
   
    for p in processes:
        p.start()
        #print("start:",p)

    for p in processes:
        p.join()
        

    
else:
    print("name_ERROR")

tf = time.time()

print("successfull, it takes time", tf-ti)

