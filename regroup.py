import numpy as np
import time

N = 40000
M = 5 
sequence = 40
correlation = 20

t_i = time.time()

ThetaX = []

Traj_Seq_Prob = []

for n in range(N):
    tempfile = np.load( "5/" + str(2*M) + "qubits" + str(sequence) + "sequcences" + str(correlation) + "sigma_trajectory" + str(n) + ".npz" )
    ThetaX.append( tempfile["ThetaX"] )
    Traj_Seq_Prob.append( tempfile["Seq_Prob"] )

print("all temped")
np.savez( "sum/" + str(2*M) + "qubits_" + str(sequence) + "sequcences" + str(N) + "trajectories_sigma=" + str(correlation) + "_5.npz", ThetaX = ThetaX, Traj_Seq_Prob = Traj_Seq_Prob ) 

t_f = time.time()

print("it takes ", t_f-t_i)