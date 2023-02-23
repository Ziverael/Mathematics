"""
Module provide simulation of client service center. It helps in finding parametets,that mean number of workers to avoid great number of missed calls.
There are Monte Carlo simulations for different parameters and plots representing corelation beetwen number of workers and calls.\n
As we know waiting for next call comes from exponential distribution with intensity lambda and expected value is 1/lambda, so we will manipulate that parameter also.\n
However, finally intensity should be related with data representing average time of waiting.
"""
import numpy as np
import random
from numba import jit
from math import log, sin, pi
import scipy
import scipy.optimize
import seaborn as sns
import matplotlib.pyplot as plt

###Constans###
MAX_ITERS = 100
#Represent Call and end of ring
CALL = True
END_OF_RING = False

def figure_setting_on() -> None:
    sns.set_theme()
    sns.set(rc={'figure.figsize':(11.7 * 2,8.27 * 2)})
    sns.set(font_scale = .8)


class BinaryHeap():
    def __init__(self):
        self.heap_list = [None]
        self.current_size = 0

    def __perc_up__(self, i):
        while i // 2 > 0:
            if self.heap_list[i] < self.heap_list[i // 2]:
                tmp = self.heap_list[i // 2]
                self.heap_list[i // 2] = self.heap_list[i]
                self.heap_list[i] = tmp
            i = i // 2
    
    def __prec_down__(self, i = 1):
        while i * 2 <= self.current_size:
            mc = self.__min_child__(i)
            if self.heap_list[i] > self.heap_list[mc]:
                tmp = self.heap_list[i]
                self.heap_list[i] = self.heap_list[mc]
                self.heap_list[mc] = tmp
            i = mc
    
    def __min_child__(self, i):
        if i * 2 + 1 > self.current_size:
            return i * 2
        else:
            if self.heap_list[i * 2] < self.heap_list[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1
    
    def insert(self, element):
        self.heap_list.append(element)
        self.current_size += 1
        self.__perc_up__(self.current_size)
    
    def find_min(self):
        return self.heap_list[1]

    def find_max(self):
        return max(self.heap_list)
    
    def pop(self):
        out  = self.heap_list[1]
        self.heap_list[1] = self.heap_list[self.current_size]
        self.current_size -= 1
        self.heap_list.pop()
        self.__prec_down__()
        return out
    
    def is_empty(self):
        return self.current_size == 0
    
    def size(self):
        return self.current_size
    
    def build_heap(self, alist):
        self.current_size = len(alist)
        self.heap_list = [None] + alist[:]
        i = self.current_size // 2
        while i > 0:
            self.__prec_down__(i)
            i -= 1
        
    def __str__(self):
        return "{}".format(self.heap_list[1:])
    
    def __repr__(self):
        return self(str)



class Event():
    """
    Event is a call or end of a ring.\n
    Attributes
    ----------
    type    [boolean]   if call set True if end of ring set False\n
    time    [float]     time when event occurs passed in sec from beggining of work\n

    """
    def __init__(self, type : bool, time : float) -> None:
        self.__type = type
        self.__time = time
    
    def get_time(self) -> float:
        return self.__time
    
    def __str__(self) -> str:
        if self.__type:
            text = "CALL"
        else:
            text = "END OF A RING"
        return text + "AT {} SEC"
    
    def __repr__(self) -> str:
        """
        Return event representation
        """
        return str(self)
    
    def __eq__(self, other) -> bool:
        """
        For == operator:\n
        if other is Event then compare times of occurance\n
        if other is float or int then compare with time of occurance
        """
        if type(self) == type(other):
            return self.__time == other.__time
        elif type(other) in (type(1), type(.1)):
            return self.__time == other
        elif type(other) == type(True):
            return self.__type == other
        else:
            raise TypeError("Cannot compare with type {}".format(other))
    
    def __gt__(self, other) -> bool:
        if type(self) == type(other):
            return self.__time > other.__time
        elif type(other) in (type(1), type(.1)):
            return self.__time > other
        else:
            raise TypeError("Cannot compare with type {}".format(other))
    
    def __lt__(self, other) -> bool:
        if type(self) == type(other):
            return self.__time < other.__time
        elif type(other) in (type(1), type(.1)):
            return self.__time < other
        else:
            raise TypeError("Cannot compare with type {}".format(other))
    
    def __ge__(self,other) -> bool:
        return not self < other

    def __le__(self, other) -> bool:
        return not self > other
    
    def __add__(self, other) -> float:
        if type(self) == type(other):
            return self.__time + other.__time
        elif type(other) in (type(1), type(.1)):
            return self.__time + other

class Simulation():
    """
    Simulation of day work in client service center.\n
    Attributes
    -----------
    finish  [int]   length of work in seconds\n
    workers [int]   positive number of workers\n
    calls   [list]  list of floating points representing call events

    """
    
    @staticmethod
    @jit(nopython = True) # check if jit can be applied in class
    def pp(t : float, lbd : float) -> list:
        """
        Return steps moments from poisson process.\n
        If time is to short then may happen that no step moment appears.
        Arguments
        ---------
        t   [float] length of poisson process\n
        lbd [float] intensity of poisson process\n
        Return
        -----------
        steps   [list]  list of floating points representing moments of poisson process
        """
        ts = -log(random.random()) / lbd
        out = []
        while ts <= t:
            out.append(ts)
            ts -= log(random.random()) / lbd
        return out

    @staticmethod
    #@jit(nopython = True)  #Try if you can
    def npp(lbd_fun, t : float, lbd : float) -> list:
        """
        Return steps moments from nonhomogeneous poisson process. Get simulation from thinnig method.\n
        Arguments
        ----------
        t   [float] length of poisson process\n
        lbd [float] intenisty of poisson process\n
        lbd_fun [function]  intensity function of nonhomogeneous poisson process\n
        Return
        ----------
        calls   [list] list of calls Events
        """
        ts = - log(random.random()) / lbd
        out = []
        while ts <= t:
            if random.random() <= lbd_fun(ts) / lbd:
                out.append(Event(CALL, ts))
            ts -= log(random.random()) / lbd
        return out
    
    @staticmethod
    @jit(nopython = True)
    def rand_exp(intensity : float):
        if intensity > 0:
            return - log(random.random()) / intensity
        else:
            raise ValueError("intensity is a positive number")
    

    def __init__(self, calls_intensity_function, workers : int, finish_time : int, mean_call_length : int = 1/(5 * 60)) -> None:
        self.__finish = finish_time
        self.__workers = workers
        self.__avg_call_length = mean_call_length
        self.__lbd_fun = jit(calls_intensity_function, nopython = True)
        #Compute lbd value
        self.__lbd = calls_intensity_function(scipy.optimize.fmin(lambda x: -calls_intensity_function(x), 1, disp = False)[0])
        buffer =  Simulation.npp(self.__lbd_fun, self.__finish, self.__lbd)
        iter = 0
        while not buffer and iter < MAX_ITERS:
            buffer =  Simulation.npp(self.__lbd_fun, self.__finish, self.__lbd) # if no calls generated repeat simulation
            iter += 1
        if iter >= MAX_ITERS:
            raise ValueError("For given t value and intenisty function cannot generate steps moments.")
        
        self.__calls =  buffer
        self.__missing_calls = 0
        self.__calls_ends = []
        self.__calls_lengths = []
        self.__missing_calls_moments = []
        self.simulation(False)
    
    def simulation(self, new_calls_events : bool = True):
        """
        That simulation is independent from call moments because they were generate earlier.
        """
        #Reset varibales storing informations about calls
        self.__missing_calls = 0
        self.__missing_calls_moments = []
        self.__calls_ends = []
        self.__calls_lengths = []

        if new_calls_events:
            buffer =  Simulation.npp(self.__lbd_fun, self.__finish, self.__lbd )
            iter = 0
            while not buffer and iter < MAX_ITERS:
                buffer =  Simulation.npp(self.__lbd_fun, self.__finish, self.__lbd) # if no calls generated repeat simulation
                iter += 1
            if iter >= MAX_ITERS:
                raise ValueError("For given t value and intenisty function cannot generate steps moments.")
            self.__calls =  buffer
        
        waiting_worekrs = self.__workers #Ready for get a call
        event_heap = BinaryHeap()
        event_heap.build_heap(self.__calls)
        ts = 0
        calls_counter = 0   #calls in total
        while not event_heap.is_empty() or calls_counter < len(self.__calls):
            current_event = event_heap.pop()
            if current_event == CALL:
                """
                If event is a call reduce number of ready worker
                and if there is no ready worker add new missing call
                """
                waiting_worekrs -= 1
                calls_counter += 1
                if waiting_worekrs < 0:
                    waiting_worekrs = 0
                    self.__missing_calls += 1
                    self.__missing_calls_moments.append(current_event.get_time())
                    continue
                """
                Simulate a  moment of talk finish (length of talk + talk start) ONLY IF call is not missing.
                """
                
                self.__calls_lengths.append(Simulation.rand_exp(self.__avg_call_length))
                end_call = Event(END_OF_RING, current_event + self.__calls_lengths[-1])
                self.__calls_ends.append(end_call)
                event_heap.insert(end_call)

            else:
                waiting_worekrs += 1
                


    def get_calls(self) -> list:
        return self.__calls
    
    def get_workres_number(self) -> int:
        return self.__workers
    
    def get_lambda(self) -> float:
        return self.__lbd

    def get_missing_calls(self) -> int:
        return self.__missing_calls

    def get_plots(self):
        fig = plt.figure()
        fig.add_subplot(2,2,1)
        #Call moments plot
        x = [i.get_time() for i in self.__calls]
        x = [0] + x + [self.__finish]
        y = np.arange(len(x))
        y[-2] = y[-1]
        sns.lineplot(x = x, y = y, drawstyle = "steps-post")
        plt.title("Incoming calls")
        plt.xlabel("time [s]")
        plt.ylabel("Incoming calls")

        #Ready workers
        fig.add_subplot(2,2,2)
        i = 1
        event_heap = BinaryHeap()
        event_heap.build_heap(self.__calls + self.__calls_ends)
        y = np.ndarray(event_heap.size() + 1, dtype = int)
        x = np.ndarray(event_heap.size() + 1, dtype = int)
        y[0] = self.__workers
        x[0] = 0
        while not event_heap.is_empty():
            event = event_heap.pop()
            x[i] = event.get_time()
            if event == CALL:
                if y[i - 1] <= 0:
                    y[i] = 0
                else:
                    y[i] = y[i - 1] - 1
            else:
                y[i] = y[i - 1] + 1
            i += 1
        sns.lineplot(x = x, y = y, drawstyle = "steps-post")
        plt.title("Ready workers")
        plt.xlabel("time [s]")
        plt.ylabel("Ready workers")

        fig.add_subplot(2,2,3)
        y = [i for i in range(self.__missing_calls + 2)]
        y[-1] = y[-2]
        sns.lineplot(x = [0] + self.__missing_calls_moments + [self.__finish],
        y = y,
        drawstyle = "steps-post")
        plt.title("Missing calls")
        plt.xlabel("time[s]")
        plt.ylabel("Missing calls")
        fig.add_subplot(2,2,4)
        sns.histplot(x = self.__calls_lengths, stat = "density")
        plt.title("Calls lengths")
        plt.xlabel("time [s]")
        plt.ylabel("Percent [%]")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    T = 28800 #s
    workers = 20
    intensity = lambda t : 0.02*sin(t * pi / (T *1.05))
    center = Simulation(intensity, workers, T)
    figure_setting_on()
    center.get_plots()