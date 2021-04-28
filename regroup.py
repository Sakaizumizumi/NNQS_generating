import numpy as np
import time


M = 5  # number of pairs of qubits
sequence = 40 # number of sequence 
correlation = 20 # correlation length
N = 40000 # number of trajectories

t_i = time.time()

ThetaX = []

Traj_Seq_Prob = []

for n in range(N):
    tempfile = np.load(  str(2*M) + "qubits" + str(sequence) + "sequcences" + str(correlation) + "sigma_trajectory" + str(n) + ".npz" )
    ThetaX.append( tempfile["ThetaX"] )
    Traj_Seq_Prob.append( tempfile["Seq_Prob"] )

print("all temped")
np.savez( str(2*M) + "qubits_" + str(sequence) + "sequcences" + str(N) + "trajectories_sigma=" + str(correlation) + "_5.npz", ThetaX = ThetaX, Traj_Seq_Prob = Traj_Seq_Prob ) 

t_f = time.time()

print("it takes ", t_f-t_i)