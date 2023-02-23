####################PREPROCESSING###########
import simulation as sim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numba import jit
from math import exp
from sys import exit
####################FUNKCJE#################
#Mała modyfikacja: zakładamy J = 1
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

        dE = 2  * s * sum(s_ngs) # assumption J = 1
        
        if dE <= 0:
            tab[i][j] = s * -1

        elif np.random.rand() < exp(-dE/t):
            tab[i][j] = s * -1


#Modyfikacja: Zwróć finalną konfigurację spinów
@jit(nopython = True)
def simulation(step_func, mcs : int, size : int, temp : float , jval : float = 1, kb : float = 1, sorted : bool = False):
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
    return e_mean, tab

############################################

##################MAIN######################
sns.set_theme()
sizes = np.array([10, 50, 100])
mcs = 10000
jval = 1
termalization = 5000

#Zapisuj dane do pliku tekstowego
with open('ising.txt', 'w') as f:
    """
    Konfigurację spinów dla L = 10 i L = 100 dla T ∗ = KB T /J = 1 < T ∗c , T ∗ = 2.26 ≈
    T ∗
    c , T ∗ = 2.26 ≈ T ∗
    c , T ∗ = 4 > T ∗
    c .
    """
    f.write("Podunkt 1\n")
    f.write("Start uporządkowany: 1\n")
    t = np.array([1, 2.26, 4])
    for k in sizes:
        data = [simulation(update_v3, mcs, k, i, 1, 1, True) for i in t]
        mean = [sum(np.abs(i[0][termalization:])) / ( mcs - termalization) for i in data]
        for i in range(3):
            f.write("L={}; T*={}".format(k, t[i]) + '\n')
            for j in data[i][0]:
                f.write(str(j)+'\n')
            f.write("\n")
            f.write("Magnetization: " + str(mean[i]))
            f.write("\n")
            sns.heatmap(data[i][1], square = True, cbar = True, cmap = ['#003049',  '#219ebc'])
            plt.title("Konfiguracja spinów dla L={}, T*={}, uporządkowane 1".format(k, t[i]))
            plt.tick_params(
                labelbottom = False,
                labelleft = False 
            )
            plt.savefig("spins L = {} t = {} sorted.pdf".format(k, t[i]), format = 'pdf')
            plt.clf()

    f.write("Start nieuporządkowany:\n")
    t = np.array([1, 2.26, 4])
    for k in sizes:
        data = [simulation(update_v3, mcs, k, i, 1, 1, False) for i in t]
        mean = [sum(np.abs(i[0][termalization:])) / ( mcs - termalization) for i in data]
        for i in range(3):
            f.write("L={}; T*={}".format(k, t[i]) + '\n')
            for j in data[i][0]:
                f.write(str(j)+'\n')
            f.write("\n")
            f.write("Magnetization: " + str(mean[i]))
            f.write("\n")
            sns.heatmap(data[i][1], square = True, cbar = True, cmap = ['#003049',  '#219ebc'])
            plt.title("Konfiguracja spinów dla L={}, T*={} nieuporządkowane".format(k, t[i]))
            plt.tick_params(
                labelbottom = False,
                labelleft = False 
            )
            plt.savefig("spins L = {} t = {} unsorted.pdf".format(k, t[i]), format = 'pdf')
            plt.clf()
    f.write('\n')
    
    #Pojedyncze trajektorie dla temperatur: T ∗ = 1.7 dla każdego L (3 osobne rysunki)
    f.write("Podunkt 2 (start uprządkowany)\n")
    t = 1.7
    data = [sim.simulation(update_v3, mcs, i, t, 1, True) for i in sizes]
    mean = [sum(np.abs(i[termalization:])) / ( mcs - termalization) for i in data]

    for i in range(3):
        plt.plot(np.arange(1,mcs + 1), data[i])
        plt.xlim([0, mcs])
        plt.ylim([-1, 1])
        plt.title("Trajektoria średniej magnetyzacji dla L={}, T*={} ".format(sizes[i], t))
        plt.xlabel("t[MCS]")
        plt.ylabel("m(t)")
        plt.savefig('Trajectory L = {}.pdf'.format(sizes[i]), format = "pdf")
        plt.clf()
        for j in data[i]:
            f.write(str(j)+"\n")
        f.write("\n")
        f.write("Magnetization: " + str(mean[i]))
        f.write("\n")
    #Magnetyzację jako funkcję temperatury dla zakresu temperatur T ∈ (1, 3.5). Tu
    #wszystkie wielkości sieci na jednym rysunku i legenda dla L
    
    t = np.linspace(1, 3.5, 50, endpoint = True)
    describ = []
    for i in sizes:
        data = [sim.simulation(update_v3, mcs, i, j, 1, 1, True) for j in t]
        """
        fig = plt.figure()
        #Wykresy kolejnych trajetorii
        for i in range(len(t)):
            ax = fig.add_subplot(5,6,i + 1)
            sns.lineplot(x = np.arange(1,mcs + 1), y = data[i])
        plt.xlim([0, mcs])
        plt.ylim([-1, 1])
        plt.xlabel("t[MCS]")
        plt.ylabel("m(t)")
        plt.savefig('Trajectiories_for_following_vals L={}.pdf'.format(i), format = 'pdf')
        plt.clf()
        """
        magnetization = [sum(np.abs(j[termalization:])) / ( mcs - termalization) for j in data]
        magnetization_squared = [sum(j[termalization : ] * j[termalization : ]) / (mcs - termalization) for j in data]
        chi = [i * i * (magnetization_squared[j] - magnetization[j] ** 2) / t[j] for j in range(len(t))]
        describ.append({
            'mean' : data,
            'magnetization' : magnetization,
            'unresist' : chi
        })
        for j in data:
            for string in j:
                f.write(str(string) + '\n')
                f.write('\n')
    #Plots
    plt.title(" <m>(T*)")
    plt.xlabel("T*")
    plt.ylabel("<m>")
    for i in range(3):
        sns.scatterplot(x = t, y = describ[i]['magnetization'] )
        sns.lineplot(x = t, y = describ[i]['magnetization'], label = "L={}".format(sizes[i]))
    plt.savefig('Magnetization .pdf', format = "pdf")
    plt.clf()

    plt.title(" <m>(T*)")
    plt.xlabel("T*")
    plt.ylabel("<m>")
    #print(describ[2])
    for i in range(3):
        sns.scatterplot(x = t, y = describ[i]['unresist'] )
        sns.lineplot(x = t, y = describ[i]['unresist'], label = "L={}".format(sizes[i]))
    plt.savefig('Unresistance .pdf', format = "pdf")
