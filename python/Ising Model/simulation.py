#Preprocessing
from operator import length_hint
import numpy as np
from math import exp
import seaborn as sns #for plots
import time #for counting time
from numba import jit   #to speed up simulation
import matplotlib.pyplot as plt 
import matplotlib.animation as anim #for animating heatmap
from sys import argv    # for managing from command line

###SIMULATION###
#One step of simulation
#in four different approaches

def update_v1(tab : np.ndarray, t: float) -> None:
    L = tab.shape[0]
    for i, j in np.random.randint(0, L, (L * L, 2)):
        k = np.random.rand()
        #i,j - indices
        #k - cond
        #s = (s(j,i-1), s(j,i+1), s(j-1,i), s(j+1,i))
        s = tab[i][j]
        s_ngs = (
            tab[(i-1) % L][j],
            tab[(i+1) % L][j],
            tab[i][(j-1) % L],
            tab[i][(j+1) % L]
            )

        dE = 2 * jval * s * sum(s_ngs)
        tab[i][j] = -s + ((dE > 0) and (exp(-dE/t) <= k)) * 2 * s

def update_v2(tab : np.ndarray, t: float) -> None:    
    L = tab.shape[0]
    for i, j in np.random.randint(0, L, (L * L, 2)):
        #i,j - indices
        #k - cond
        #s = (s(j,i-1), s(j,i+1), s(j-1,i), s(j+1,i))
        s = tab[i][j] # select spin
        s_ngs = (
            tab[(i-1) % L][j],
            tab[(i+1) % L][j],
            tab[i][(j-1) % L],
            tab[i][(j+1) % L]
            ) # select its neighbours

        dE = 2 * jval * s * sum(s_ngs) #comupte dE
        """
        msg = "-" * 10 + "/"" 
        XX\t{}\tXX
        {}\t{}\t{}
        XX\t{}\tXX\n
        "/"".format(s_ngs[2], s_ngs[0], s, s_ngs[1], s_ngs[3]) + "dE: {}\n".format(dE) + "-" * 10
        print(msg)
        """


        if dE <= 0:
            tab[i][j] = s * -1

        elif np.random.rand() < exp(-dE/t):
            tab[i][j] = s * -1

@jit(nopython = True)
def update_v3(tab : np.ndarray, t: float) -> None:
    L = tab.shape[0]
    for i, j in np.random.randint(0, L, (L * L, 2)):

        #i,j - indices
        #k - cond
        #s = (s(j,i-1), s(j,i+1), s(j-1,i), s(j+1,i))
        s = tab[i][j]
        s_ngs = (
            tab[(i-1) % L][j],
            tab[(i+1) % L][j],
            tab[i][(j-1) % L],
            tab[i][(j+1) % L]
            )

        dE = 2 * jval * s * sum(s_ngs)
        
        if dE <= 0:
            tab[i][j] = s * -1

        elif np.random.rand() < exp(-dE/t):
            tab[i][j] = s * -1


@jit(nopython = True)
def update_v4(tab, t: float):
    for i, j in np.random.randint(0, L, (L * L, 2)):
        k = np.random.rand()
        #i,j - indices
        #k - cond
        #s = (s(j,i-1), s(j,i+1), s(j-1,i), s(j+1,i))
        s = tab[i][j]
        s_ngs = (
            tab[(i-1) % L][j],
            tab[(i+1) % L][j],
            tab[i][(j-1) % L],
            tab[i][(j+1) % L]
            )

        dE = 2 * jval * s * sum(s_ngs)
        tab[i][j] = -s + ((dE > 0) and (exp(-dE/t) <= k)) * 2 * s

#Complete simulation
@jit(nopython = True)
def simulation(step_func, mcs : int, size : int, temp : float , jval : float = 1, kb : float = 1, sorted : bool = False):
    """
    Return
    ------
    [np.ndarray] means od spins value in following monte carlo steps of simulation
    """
    temp = temp * kb
    total_size = size * size
    if sorted:
        tab = np.ones((size, size))
    else:
        tab = np.random.choice(np.array((-1.0, 1.0)),(size, size))
    e_mean = np.zeros(mcs)
    for i in range(mcs):
        step_func(tab, temp)
        e_mean[i] = np.sum(tab) / total_size
    return e_mean

#################In progress
"""
def sim_animation(step_func, mcs : int, size : int, temp : float , jval : float = 1, kb : float = 1, sorted : bool = False):
    temp = temp * kb
    total_size = size * size
    if sorted:
        tab = np.ones((size, size))
    else:
        tab = np.random.choice(np.array((-1.0, 1.0)),(size, size))
    e_mean = np.zeros(mcs)
    for i in range(mcs):
        if i % 1000 == 0:
            sns.heatmap(tab)
            plt.savefig('frame{}.jpg'.format(i))
        step_func(tab)
        e_mean[i] = np.sum(tab) / total_size
    return e_mean
"""
def update_speed_test(L):
    #Prepeare dataset
    tab = np.random.choice((-1, 1), (L, L))
    n = 10000
    #Precompiling with numba
    update_v3(tab)
    update_v4(tab)

    print("Testing...\ncall functions {} times for  matrix {}x{}\n-----------------".format(n,L,L))
    delta = time.time()
    for i in range(n):
        update_v1(tab)
    delta = time.time() - delta
    print("Without ifs {}".format(delta))

    delta = time.time()
    for i in range(n):
        update_v2(tab)
    delta = time.time() - delta
    print("With ifs {}".format(delta))
    
    delta = time.time()
    for i in range(n):
        update_v3(tab)
    delta = time.time() - delta
    print("With ifs and numbda {}".format(delta))
    
    delta = time.time()
    for i in range(n):
        update_v4(tab)
    delta = time.time() - delta
    print("Without ifs and numbda {}".format(delta))


#ANIMATION
def init_cond():
    sns.heatmap(tab, square = True, cbar = True, cmap = ['#003049',  '#219ebc'])

def animate(i):
    update_v3(tab, t)
    sns.heatmap(tab, square = True, cbar = False, cmap = ['#003049',  '#219ebc'])




if __name__ == "__main__":
    #call progam anim_flag test_flag L t monte_carlo_steps
    anim_flag = False
    test_flag = False
    #parsing arguments
    if len(argv) >= 3:
        anim_flag = int(argv[1])
        test_flag = int(argv[2])
    ###VARIABLES###
    L = 100
    jval = 1
    kb = 1
    #t = np.arange(.4, 10.1, .2)
    #t = np.arange(1.0, 2, .2)
    t = np.array([1.2 for i in range(30)])
    #t = np.linspace(1, 3.5, 30, endpoint = True)
    one_pic = True
    t = kb * t
    tab = np.random.choice((-1, 1), (L, L))
    monte_carlo_steps = 10000

    #parsing arguments
    if len(argv) >= 4:
        L = int(argv[3])
    if len(argv) >= 5:
        t = float(argv[4])
    if len(argv) == 6:
        monte_carlo_steps = int(argv[5])

    if test_flag:
        update_speed_test(L)
    
    if anim_flag:
        fig = plt.figure()
        ###Speed of simulation
        delta = time.time()
        animation = anim.FuncAnimation(fig, animate, init_func = init_cond, repeat = False, frames = monte_carlo_steps)
        #Save animation
        writergif = anim.PillowWriter(fps = 20) #set writer and fps number
        animation.save('Ising.gif', writer = writergif) #save with given writer
        delta = time.time() - delta
        plt.close()
    #Single simulation
    if type(t) == type(1.0):
        delta2 = time.time()
        e_means = simulation(update_v3, monte_carlo_steps, L, t, sorted = True)
        delta2 = time.time() - delta2
        plt.plot(np.arange(1,monte_carlo_steps + 1), e_means)

    else:
        delta2 = time.time()
        trajectories = [simulation(update_v3, monte_carlo_steps, L, i, sorted = False) for i in t]
        delta2 = time.time() - delta2
        if not one_pic:
            fig = plt.figure()
            for i in range(len(t)):
                ax = fig.add_subplot(5,6,i + 1)
                sns.lineplot(x = np.arange(1,monte_carlo_steps + 1), y = trajectories[i])
        else:
            for i in range(len(t)):
                plt.plot(np.arange(1,monte_carlo_steps + 1), trajectories[i])
        for i in trajectories:
            print("Magnetization: {}".format(sum(np.abs(i[5000:])) / 5000))
        print("Grouped mean: {}".format(sum([abs(i[-1]) for i in  trajectories]) / len(trajectories)))
    plt.xlim([0, monte_carlo_steps])
    plt.ylim([-1, 1])
    plt.title("Trajektorie średnich magnetyzacji T={}".format(t[0]))
    plt.xlabel("t[MCS]")
    plt.ylabel("m(t)")
    plt.show()
    if monte_carlo_steps >= 2000:
        termalization = 1000
    else:
        termalization = 800
    ##Means

    #print stats
    if anim_flag:
        print("Total animation time: {} s for size: {}x{} and {} MCS".format(delta, L, L, monte_carlo_steps))
    print("Total simulation time: {} s for size: {}x{} and {} MCS".format(delta2, L, L, monte_carlo_steps))