###PREPROCESSING###
import numpy as np
import sys
from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns
import time
from math import ceil
from random import random

class InputError(Exception):
    def __init__(self, msg = "Invalid input"):
        super().__init__(msg)


@jit(nopython = True)
def update(s : np.ndarray) -> None:
    """
    Update phase space. This is 1 Monte Carlo step for voter model simulation.\n
    Assume that input is correct.\n
    Arguments
    ---------
    s   [np.ndarray] one-dimensional numpy array of boolean values\n
    Return
    ---------
    None

    """
    n = s.shape[0]
    k = np.random.randint(0, n, n)
    for i in k:
        neigh = (
            s[(i - 1) % n],
            s[(i + 1) % n]
            )
        if random() <= 0.5:
            s[i] = neigh[0]
        else:
            s[i] = neigh[1]

@jit(nopython = True)
def simulation(update, n : int, yes : int) -> int:
    """
    Simulate Voter Model with Monte Carlo method.\n
    Assume that input is correct.\n
    Arguments
    ---------
    n   [int]   number of agents\n
    yes [int]   number of agents which agree with thesis at the beginning of simulation\n
    update    [function]  function for a phase space update\n
    Return
    ----------
    steps   [tuple]   number of steps by the consensus time and consensus status( for positive opinion 1 else 0)
    """
    agents = np.zeros(n)
    yess = np.random.permutation(n)[:yes] #Select indices of yes
    agents[yess] = 1 # Set beggining opinion to group of agents

    steps = 0

    while sum(agents) not in (0, n): #loop keep going as long as there is not consensus
        update(agents)
        steps += 1


    return steps, agents[0]
    
def simulation_vizualization(n : int, yes : int):
    """
    Simulate Voter Mode with Monte Carlo method.\n
    Assume that input is correct.\n
    Arguments
    ---------
    n   [int]   number of agents\n
    yes [int]   number of agents which agree with thesis at the beginning of simulation\n
    Return
    -------
    changes    []   list of phase spaces in  following Monte Carlo steps


    """
    agents = np.zeros(n, dtype = bool)
    yess = np.random.permutation(n)[:yes] #Select indices of yes
    agents[yess] = True # Set beggining opinion to group of agents
    pass
    

###MAIN###
if __name__ == "__main__":
    ###PARSE ARGUMENTS AND CHECK CONDITIONS###
    #Arguments: N, dx, L
    args = sys.argv[1:]
    if len(args) < 3:
        raise InputError("Passed {} arguments. Expected 3".format(len(args)))
    try:
        N = int(args[0])
        dx = float(args[1])
        L = int(args[2])
    except:
        raise TypeError("At least one of the given arguments cannot be present as a number and especially N and L must be integers.")
    
    verbose = False

    if len(args) > 3:
        if args[3] == "-v":
            verbose = True

    if N <= 0:
        raise ValueError("N must be positive")
    if not  0 < dx <= 1:
        raise ValueError("dx must be from interval (0,1]")
    if L <= 0:
        raise ValueError("L must be positive")

    ###SIMULATE###
    k = ceil(1/dx)
    densities = np.array([i*dx for i in range(k)] + [1])    
    beg_with_positive_opinion = np.ceil(densities * N)
    
    #empty arrays for P+ and time related with x
    positive_opinon = np.ndarray(k + 1) 
    steps = np.ndarray(k + 1)


    if verbose:
        print("Positive opinion density vector:{}\nAgents with positive opinion vector:{}".format(densities, beg_with_positive_opinion))
    
    with open("N{}dx{}L{}.txt".format(N, dx, L),"w") as f:
        for i in range(0,len(densities)):
            if verbose:
                print("Run simulation for density {} ...".format(densities[i]))
            delta = time.time()
            times, opinion = np.array([simulation(update, N, beg_with_positive_opinion[i]) for _ in range(L)]).T #list of steps for L repetiotions for given x
            delta = time.time() - delta
            if verbose:
                print("Finished {} simulations for given density. Total time :{} seconds".format(L, delta))
            
            steps[i] = sum(times) / L
            positive_opinon[i] = sum(opinion) / L
            
            if verbose:
                print("Mean consensus time: ", steps[i])
                print("Positive consensus: ", positive_opinon[i])
                print("-" * 10)
            
            #Write to txt file
            out_str = str(densities[i]) + "  " + str(positive_opinon[i]) + "  " + str(steps[i])
            f.write(out_str + '\n')
    
    #Make plots
    sns.set_theme()
    sns.scatterplot(\
    x = densities,
    y = steps
    )
    plt.title("Ilość potrzebnych kroków do osiągnięcia konsensusu")
    plt.xlabel("x")
    plt.ylabel("$<\tau>$")
    plt.savefig("N{}dx{}L{}_timeplot.pdf".format(N, dx, L), format = "pdf")
    plt.clf()
    print(positive_opinon)
    sns.set_theme()
    sns.scatterplot(\
    x = densities,
    y = positive_opinon
    )
    plt.title("Relacja prawdopodobieństwa pozytywnej opinii do gęstości początkowej pozytywnej opinii")
    plt.xlabel("x")
    plt.ylabel(r"$P_+$")
    plt.ylim((-0.1, 1.1))
    plt.savefig("N{}dx{}L{}_probabplot.pdf".format(N, dx, L), format = "pdf")


            