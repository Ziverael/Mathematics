"""
© 2023 Ziverael

QUEUES MODULE
Moldule provides elements for building queuing systems and simulate flow in built systems.
The idea for the module was the project for Mathematics in Industry course at university. My goal was to build API for creating queuing systems and analyse how it performs according to mathemathics laws and computations.

HOW TO USE IT?
The basic element is System class that let you build a system graph representnig queues and connections between. When you build network you can run simulaton with system method run(T). Basicly first node will be certain arrival process. You can build it like this.
system_ = System(intensity_of_arrival_process)
I also provided support for naming vertices to avoid mess while the graph expands. In default every vertex representing queue or arrival process or endpoint has it own id, which you can use, but as I said it will cause mess in bigger graphs, so I strongly recommend using names. You can do it with
system_ = System(intensity_of_arrival_process, True, "FirstVertexName")
It is crucial to point out that, you cannot simultanoulsy use indexes or vertecies names. If you decide for one you have to stick to them.
To add new objects you will use 
system_.add_node(prev_node_id_or_name, object, probability, node_name)
The object will be Queue object or None if it is endpoint. You can create few endpoints however for purity it is recommended to create one endpoint and link all ending queues to them.

TO DO
-use fliter and filterfalse for code imporvement
-add statistics methods for analysis
-better display for system, jobs etc
-graph representation and giphs
-add subclasses for system that let you build open and closed systems.
"""
###IMPORT MODULES###
import numpy as np
import random
from numba import jit
from math import log, sin, pi
import scipy
import scipy.optimize
import seaborn as sns
import matplotlib.pyplot as plt

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


###Variables###
MAX_ITERS = 100
INF = "INF"

BIRTH = "BIRTH"
ARRIVAL = "ARRIVAL"
MISSING = "MISSING"
WAITING = "WAITING"
PROCESSING = "PROCESSING"
LEAVING = "LEAVING"
CLOSED = "CLOSED"



###Supporting classes###
class QueueDataStructure():
    """
    Klasa implementująca kolejkę za pomocą pythonowej listy tak,
    że początek kolejki jest przechowywany na początku listy.
    """  
    def __init__(self):
        self.list_of_items = []

    def enqueue(self, item):
        self.list_of_items.append(item)


    def dequeue(self):
        return self.list_of_items.pop(0)

    def is_empty(self):
        return self.list_of_items == []

    def size(self):
        return len(self.list_of_items)

    def __str__(self):
        queue = "END\n"
        for item in self.list_of_items[::-1]:
            queue += str(item) + "\n"
        queue += "BEGINING"
        return queue

    def __repr__(self):
        return str(self)
    
    def find_element(self, el):
        try:
            return self.list_of_items.index(el)
        except:
            return -1
    


class BinaryHeap():
    def __init__(self):
        self.heap = [None]
        self.size = 0
    
    def __perc_up__(self, i):
        while i // 2 > 0:
            if self.heap[i] <  self.heap[i // 2]:
                tmp = self.heap[i // 2]
                self.heap[i // 2] = self.heap[i]
                self.heap[i] = tmp
            i = i // 2

    def __perc_down__(self, i = 1):
        while i * 2  <= self.size:
            mc = self.__min_child__(i)
            if self.heap[i] > self.heap[mc]:
                tmp = self.heap[i]
                self.heap[i] = self.heap[mc]
                self.heap[mc] = tmp
            i = mc

    def __min_child__(self, i):
        if i * 2 + 1 > self.size:
            return i * 2
        else:
            if self.heap[i * 2] < self.heap[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1
    
    def insert(self, element):
        self.heap.append(element)
        self.size += 1
        self.__perc_up__(self.size)
    
    def get_min(self):
        return self.heap[1]
    
    def get_max(self):
        return max(self.heap)
    
    def pop(self):
        out = self.heap[1]
        self.heap[1] = self.heap[self.size]
        self.size -= 1
        self.heap.pop()
        self.__perc_down__()
        return out
    
    def is_empty(self):
        return self.size == 0
    
    def get_size(self):
        return self.size

    def build_heap(self, alist : list):
        self.size = len(alist)
        self.heap = [None] + alist[:]
        i = self.size // 2
        while i > 0:
            self.__perc_down__(i)
            i -= 1
    
    def __str__(self):
        return "{}".format(self.heap[1:])
    
    def __repr__(self):
        return str(self)



class Edge:
    """Classs representing connections beetwen components. Also here as components are recognised arrival process
    
    Attributes
    ----------
    from_   Starting vertex
    to_   Endpoint
    prob    probability of following that edge
    
    """
    
    def __init__(self, fr, to, prob = 1):
        self.from_ = fr
        self.to_ = to
        self.prob = prob

    def __str__(self):
        return "({})--[p={}]-->({})".format(self.from_.get_key(), self.prob, self.to_.get_key())    

    def __repr__(self):
        return str(self)

    def get_beg(self):
        return self.from_
    
    def get_end(self):
        return self.to_

    def get_prob(self):
        return self.prob
 
    def set_prob(self, prob):
        self.prob = prob


class EdgeRouteError(Exception):
    def __init__(self, message = "Invalid route."):
        self.message = message
        super().__init__(self.message)


class RoutingError(Exception):
    def __init__(self, message = "Queue vertex has no route."):
        self.message = message
        super().__init__(self.message)


class MathError(Exception):
    def __init__(self, message = "Forbiden math operation."):
        self.message = message
        super().__init__(self.message)


class Vert:
    """
    Vertex class.
    
    Atributes
    ---------
    key     [int]   vertex unique key
    obj    [Object]   Type of vertex. Valid objects are ArrivalProcess, Queue and NULL recognised as leaving the system
    observing_ngh [dict]   Dictionary of object whose can enter this component ???
    observed_ngh [dict]   Dictionary of object whose can enter this component ???
    routes  [dict]  Vertices which you can use to travel between vertices

    Details
    -------
    Arrival process cannot have observing neighbours.
    NULL cannot have observed neighbours.
    """
    def __init__(self, key, obj, name = "") -> None:
        if type(key) != type(0):
            raise TypeError
        self.key = key
        self.obj = obj
        self.name = name
        # self.observed_ngh = {}
        # self.observing_ngh = {}
        self.routes = {}


    def __add_route__(self, vert, prob = 1) -> None:
        """
        Add an  observed neighbour. The  neighbour is only a vertex to which you can travel
        from this vertex. Create 
        Return
        ------
        Edge object. The goal is to pass the edge to Graph object.
        """
        # k, w = vert.get_key(), vert.get_weight()
        # self.nght[vert.get_key()] = Vert(k, w)
        # self.counter += 1
        if not isinstance(vert, Vert):
            raise TypeError("vert should be Vert object or subtype object.")
        if self.obj is None:
            raise EdgeRouteError("None vertex cannot provide routes.")
        edge = Edge(self, vert, prob)
        self.routes[vert.get_key()] = edge
        return edge
    
    def get_routes(self) -> dict:
        return list(self.routes.values())
    
    def get_key(self, force_key = False) -> int:
        """Return key of the vertex. However return name if name is not empty string. If name is specified one can get key with argument force_key set to True."""
        if self.name and not force_key:
            return self.name
        return self.key

    def _check_routes_(self):
        """Check if probabilities cumulates to 1 and raise exception if not"""
        if sum([edge.get_prob() for edge in self.routes.values()]) != 1:
            raise MathError("Probabilities should sum up to value 1.")


    def __str__(self) -> str:
        if self.name:
            first_line = "NodeID: {}\tName: {}\tObject:{}".format(self.key, self.name, self.obj)
        else:
            first_line = "NodeID: {}\tObject:{}".format(self.key, self.obj)
        routes = ""
        for i in self.routes:
            routes += str(self.routes[i])
        return first_line + "\n" +  routes


    def __repr__(self) -> str:
        return str(self)
    
    def get_object_type(self):
        return type(self.obj)
    
    def choice_route(self) -> Edge:
        """"Select one Vert object from to which exist route from the Vert. If there are few routes then route will be choosen randomly with given probabilities."""
        
        if self.routes == {}:
            raise RoutingError
        elif len(self.routes) == 1:
            return list(self.routes.values())[0].get_end()
        else:
            ids, probs = zip(*self.routes.items())
            probs = [*map(lambda t: t.get_prob(), probs)]
            vert = np.random.choice(ids, p = probs)
            return self.routes[vert].get_end()


class ArrivalVertex(Vert):
    def __init__(self, key, obj, name = "", arrival_counter = 0):
        if not isinstance(obj, ArrivalProcess):
            raise TypeError("Arrival Vertex can only handle Arrival Process.")
        super().__init__(key, obj, name)
        self.arrival_counter = arrival_counter
        self.history = {
            "Events" : [],
            "Jobs" : []
        }
        self.jobs = {}
    
    def generate_arrivals(self, finish_time):
        """Generate arrivals
        
        Args
        ----
        finish time [number]    Finish time of the generation

        Return
        ------
        Tulpe of lists representing correspondingly jobs and events
        """
        
        self.obj.generate_arrivals(finish_time)
        arrs = self.obj.get_arrivals()
        arr_indexes = range(self.arrival_counter + 1, self.arrival_counter + len(arrs) + 1)
        self.arrival_counter += len(arrs)
        jobs = [Job(id_, time_, self.get_key()) for (id_, time_) in zip(arr_indexes, arrs)]
        events = [Event(id_, self.get_key(), BIRTH, time_) for id_, time_ in zip(arr_indexes, arrs)]
        self.history["Events"].append(events)
        self.history["Jobs"].append(jobs)
        self.jobs.update({id_ : job_ for id_, job_ in  zip(arr_indexes, jobs)})
        return jobs, events
    
    def get_history(self):
        """Return disctionary with archive of the events and jobs generated in that node."""
        return self.history
    
    def get_arrivals_number(self):
        return self.obj.get_arrivals_number()
    
    def get_process(self):
        return self.obj.get_process()

    def __reset_node__(self):
        """Reset node history """
        self.history = {
            "Events",
            "Jobs"
        }
        self.arrival_counter = 0
        self.obj.reset_arrivals()
    
    def _set_arrival_counter_(self, val):
        self.arrival_counter = val
    
    def set_process(self, type_):
        self.obj.set_process(type_)
    
    def get_jobs(self) -> list:
        return list(self.jobs.values())

    
    def dequeue(self, jobId_, time_):
        """
        Dequeue Job object from the ArrivalVertex.

        Args
        ----
        jobId_  [int]   Valid jobId
        
        Return
        ------
        
        """
        if jobId_ not in self.jobs:
            raise KeyError("No Job object with given id.")
        job_ = self.jobs.pop(jobId_)
        vert = self.choice_route()
        events = []
        events.append(Event(jobId_, vert.get_key(), ARRIVAL, time_))
        enq_events = vert.enqueue(job_, time_)
        if type(enq_events) == list:
            events.extend(enq_events)
        else:
            events.append(enq_events)
        # events.extend(enq_events)
        return events

        
        
        

class QueueVertex(Vert):
    """Queue vertex class
    
    It is crucial to point out that in jobs there are simple references to jobs which also should appear in the system.
    """
    
    def __init__(self, key, obj, name = ""):
        if not isinstance(obj, Queue):
            raise TypeError("Queue Vertex can only handle Queue object.")
        super().__init__(key, obj, name)
        self.jobs = {}
    
    def get_servers(self) -> list:
        """Return list og Server objects in the QueueVertex."""
        return self.obj.get_servers()
    
    def get_buffer(self):
        return self.obj.get_buffer()

    def get_intensities(self) -> list:
        return self.obj.get_intensities()
    
    def get_jobs(self) -> list:
        """Return list of jobs in the Queue. Jobs are Job objects."""
        return list(self.jobs.values())
    
    def get_jobs_at_servers(self) -> list:
        """Return list of jobs at servers. Jobs are Job objects."""
        return [self.jobs[i] for i in self.obj.get_payloads()]
    
    def get_jobs_at_buffer(self) -> list:
        """Return list of jobs in the buffer. Jobs are Job objects."""
        at_server = self.obj.get_payloads()
        at_buffer = [job for job in self.jobs.keys() if job not in at_server]
        return [self.jobs[index] for index in at_buffer]
    
    def get_empty_servers(self) -> list:
        """Return list of Server objects which are not busy."""
        indexes =  [ id_ for id_, status in self.obj.find_all_empty_servers() if status]
        return [self.obj.servers[index] for index in indexes]
    
    def get_servers_number(self) -> int:
        """"Return number of Server objects it the QueueVertex."""
        return self.obj.get_servers_number()

    def set_intensities(self, ints_):
        self.obj.set_intensities(ints_)
    
    def _set_payload(self, payload, serverId = -1):
        """"Set payload to the server. Return -1 if payload was rejected and id of the server if it was succesfull."""
        return self.obj.set_payload(payload, serverId)
    
    def enqueue(self, job_, time_):
        """Enqueu Job to the QueueVertex.
        
        Args
        ----
        job_    [Job]   Job object to enqueue
        time_   [number] Time of the occurence

        Return
        ------
        Event generated during enqueuing to that queue.
        """
        if not isinstance(job_, Job):
            raise TypeError("Only Job object could be enqueue to the vertex.")
        enqueu_status = self.obj.process(job_.get_id())
        if enqueu_status == -2:
            job_.set_status(MISSING, time_, self.get_key())
            return Event(job_.get_id(), self.get_key(), MISSING, time_)
        elif enqueu_status == -1:
            self.jobs[job_.get_id()] = job_
            job_.set_status(WAITING, time_, self.get_key())
            return Event(job_.get_id(), self.get_key(), WAITING, time_)
        else:
            self.jobs[job_.get_id()] = job_
            job_.set_status(PROCESSING, time_, self.get_key())
            events = []
            events.append(Event(job_.get_id(), self.get_key(), PROCESSING, time_))
            events.append(Event(job_.get_id(), self.get_key(), LEAVING, time_ + enqueu_status))
            return events

    def dequeue(self, jobId_, time_) -> list:
        """Dequeue job.
        That function shouldd be triggered by event. It update given job, by changing it status and pushing forward in the system.
        The method also check the buffer and set payload for the empty server. The method trigger enqueue method of the next 
        
        Exceptions
        ----------
        RoutingError
            It is raisen when there is no route from this vertex
        NameError
            It is raisen when the job in not at one of the queue're servers.
        
        Return
        ------
        List of events
        """
        host_server = None
        if jobId_ not in self.jobs:
            raise KeyError("No Job object with given id.")
        job_ = self.jobs.pop(jobId_)
        # Get server that
        for server in self.get_servers():
            if server.get_payload() == jobId_:
                host_server = server
                break
        if host_server == None:
            raise NameError("Job not founded at any server. The job is in waiting space.")
        vert = self.choice_route()
        # routes = self.get_routes()
        # if routes == {}:
        #     raise RoutingError
        # elif len(routes) == 1:
        #     vert = list(routes.values())[0]
        # else:
        #     ids, probs = zip(*routes.items())
        #     vert = np.random.choice(ids, p = probs)
        #     vert = routes[vert]
        events = []
        #  Remove job from the queue
        host_server.remove_payload()
        #Push object from waiting space to the queue
        next_jobId = self.get_buffer().dequeue()
        if next_jobId:
            host_server.set_payload(next_jobId)
            stime_ = host_server.service()
            events.append(Event(next_jobId, self.get_key(), PROCESSING, time_))
            events.append(Event(next_jobId, self.get_key(), LEAVING, time_ + stime_))
            self.jobs[next_jobId].set_status(PROCESSING, time_, self.get_key())
        # Generate events
        events.append(Event(job_.get_id(), vert.get_key(), ARRIVAL, time_))
        enq_event = vert.enqueue(job_, time_)
        #Return all events
        if type(enq_event) == list:
            events.extend(enq_event)
        else:
            events.append(enq_event)
        return events
        


class LeavingtVertex(Vert):
    """Vertex representing system exit. It is good idea to set only one LeavingVertex in a system and connect it with all finished queues.
    
    """
    def __init__(self, key, name = ""):
        super().__init__(key, None, name)
        self.jobs = {}
        self.job_counter = 0

    def enqueue(self, job_, time_):
        if not isinstance(job_, Job):
            raise TypeError("Only job object can be enqueue to LeavingVertex.")
        self.job_counter += 1
        self.jobs[job_.get_id()] = job_
        job_.set_status(CLOSED, time_, self.get_key())
        return Event(job_.get_id(), self.get_key(), CLOSED, time_)
    
    def get_jobs(self):
        return list(self.jobs.values())
    
    def __contains__(self, key : int) -> bool:
        return key in self.jobs
    
    def get_jobs_counter(self):
        return self.job_counter
    
    def __str__(self) -> str:
        jobs = ""
        for i in self.jobs.values():
            jobs += str(i) + "\n"
        return "NodeID: {}\tLeavingVert with jobs:{}".format(self.get_key(), jobs)
    


class Graph:
    """
    Graph representation.
    
    Atributes
    ---------
    verts   [dict]  list of vertices in a graph
    edges   [list]  list of edges in a graph
    starting vetex  [Vert]  Starting node for the graph.
    """
    def __init__(self, starting_vert, use_name : bool = False, starting_name : str = "Arrival"):
        if not isinstance(starting_vert, ArrivalProcess):
            raise TypeError("Starting vertex should be arrival vertex.")
        self.key_counter = 0
        if use_name:
            self.toggle_names = True
            key = starting_name
            self.starting_vert = ArrivalVertex(0, starting_vert, starting_name) 
        else:
            self.toggle_names = False
            key = 0
            self.starting_vert = ArrivalVertex(0, starting_vert) 

        self.verts = {
            key : self.starting_vert
        }
        self.edges = []

    # def __str__(self) -> str:
    #     verts = ""
    #     for i in self.verts:
    #         verts += str(self.verts[i])

    #     return  verts


    def add_vert(self, obj, prev_node_id, prob = 1, name : str = "") -> None:
        """Add new vertex. It is crucial to point out that we always define route from existing node to that node.
        
        """
        if self.toggle_names:
            #Then prev_node_id should be a string
            if not prev_node_id in self.verts or not name:
                raise ValueError("Given node do not exist or name is not specified.")
        elif prev_node_id > self.key_counter:
            raise ValueError("Given node do not exist.")
        
        self.key_counter += 1
        if type(obj) == Queue:
            new_vert = QueueVertex(self.key_counter, obj, name)
        elif obj == None:
            new_vert = LeavingtVertex(self.key_counter, name)
        else:
            raise TypeError("Invalid object type.")
        if self.toggle_names:
            self.verts[name] = new_vert
        else:
            self.verts[self.key_counter] = new_vert
        self.edges.append(self.verts[prev_node_id].__add_route__(new_vert, prob))
    
    def add_route(self, prev_node_id : int, next_node_id : int, prob : float = 1) -> None:
        if self.toggle_names:
            #Then despite node ids prev_node_id and next_node_id store strings
            if prev_node_id not in self.verts or next_node_id not in self.verts:
                raise ValueError("One of the given nodes does not exist.")
        elif prev_node_id < 0 or prev_node_id > self.key_counter or next_node_id < 0 or next_node_id > self.key_counter:
            raise ValueError("One of the given nodes does not exist.")
        self.edges.append(self.verts[prev_node_id].__add_route__(self.verts[next_node_id], prob))

    def _graph_validation_(self):
        """
        Check if the queue graph is valid.
        It is important to note this function has not applied checking if every route ends with LeavingVertex.
        """
        for edge in self.edges:
            if edge.get_beg().get_object_type() == LeavingtVertex:
                raise RoutingError("LeavingVertex cannot has any route.")
            elif edge.get_end().get_object_type() == ArrivalProcess:
                raise RoutingError("ArrivalVertex cannot be the endpoint of any route.")
        for vert in self.verts.values():
            if vert.get_routes():
                vert._check_routes_()

    def __contains__(self, key : int) -> bool:
        return key in self.verts

    def __getitem__(self, key) -> Vert:
        if key not in self.verts:
            raise ValueError
        return self.verts[key]

    def get_starting_vert(self) -> Vert:
        return self.starting_vert
    
    def get_vertices(self) -> dict:
        return self.verts
    
    def get_vertex(self, id_) -> Vert:
        """"Return vertex. If use_name set to True then use name despite id."""
        return self.verts[id_]
    
    def __get_key_counter__(self):
        return self.key_counter
    
    def __str__(self):
        verts = ""
        if self.toggle_names:
            for vert_name in self.verts:
                verts += "Name:" + vert_name + "\t" + str(self.verts[vert_name]) + "\n"
        else:    
            for vert in self.verts.values():
                verts += str(vert) + "\n"
        return "Graph with {} verices and {} routes:\n{}".format(len(self.verts), len(self.edges), verts)


    # def dot_repr(self) -> str:
    #     """
    #     Generate dot representation.
    #     """
    #     out = "digraph G{"
    #     verts = {}
    #     for i in self.edges:
    #         fr, to = i.get_beg().get_key(), i.get_end().get_key()
    #         verts[fr], verts[to] = None, None
    #         out += "{}->{};".format(fr, to)
    #     for i in self.verts:
    #         if self.verts[i].get_key() not in verts:
    #             out += " " + str(self.verts[i].get_key()) + " "
    #     out += "}"
    #     return out





###Classes###
class ArrivalProcess():
    """Class representing system arrival process
    
    Attrributes
    -----------
    process     function generating arrivals
    process_type    [srt]   string representing the name of the arrival process
    current_time    [float] Starting time for the arrival process. Process start from reference time 0 and after every generation of arrivals it will continue from the last time. Therefore we can continue simulation from previous time moment.
    arrivals    [np.array]  Array of floating points representing arrivals.
    arrivals_number [int]   total number of arrivals from the arrival process
    intensity   [float] intensity of the arrival process

    Methods
    -------
    __init__(intensity, type_ = "Poisson")

    
    """
    

    @staticmethod
    @jit(nopython = True)
    def poisson_process(t : float, lbd : float) -> list:
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

    # arrival_process = {
    #     "Poisson" : ArrivalProcess.poisson_process,
    #     "Default" : ArrivalProcess.poisson_process,
    # }


    def __init__(self, intensity, type_  = "Poisson"):
        """sumary_line
        
        Args
        ----
        arrival_counter [int]   index of the las arrival. It is used for providing unique ids for jobs, especially if there are few arrival procceses in the system.
        """
        
        
        if type_ not in arrival_process.keys():
            raise ValueError("Invalid Arrival process")
        
        # self.process = ArrivalProcess.arrival_process[type]
        self.process_type = type_
        self.process = arrival_process[type_]
        self.arrivals = []
        self.arrivals_number = 0
        self.intensity = intensity
        self.current_time = 0
    
    def generate_arrivals(self, finish_time : float):
        self.arrivals = np.array(self.process(finish_time, self.intensity))
        self.arrivals += self.current_time
        self.arrivals_number += len(self.arrivals)
        self.current_time = finish_time
        # arr_indexes = range(self.arrival_counter, self.arrival_counter + len(arr_time))
        # arrivals = [Job(id_, self.nodeId, time_) for (id_, time_) in zip(arr_indexes, arr_time)] # Generate jobs but here put eventrs
        # self.arrivals = arrivals
        #self.system.new_jobs()
    
    def get_arrivals(self):
        return self.arrivals
    
    def get_arrivals_number(self):
        return self.arrivals_number
    
    def reset_arrivals(self):
        self.arrivals = []
        self.arrivals_number = 0

    def get_process(self):
        return self.process_type
    
    def set_process(self, type_):
        if type_ not in arrival_process.keys():
            raise ValueError("Invalid Arrival process")
        
        self.process_type = type_
        self.process = arrival_process[type_]
    
    def __str__(self):
        return "Arrival Process:{} with intensity {}".format(self.process_type, self.intensity)

    def __repr__(self):
        return str(self)
        



arrival_process = {
        "Poisson" : ArrivalProcess.poisson_process,
        "Default" : ArrivalProcess.poisson_process,
    }





class Server():
    """Server class
    This abstraction layer store ids of jobs. Server can store job id as payload, generate service time and remove payload.
    
    Attributes
    ----------
    intensity
    service_time_generator  [function]  random number generator for generating service time
    payload    [int]   Job's id. If the value is 0 then no payload at the server

    Methods
    -------
    __init__

    get_intensity

    get_service_time_generator

    get_payload

    set_service_time_generator

    set_intensity

    set_payload

    is_empty

    service
    """

    @staticmethod
    @jit(nopython = True)
    def rand_exp(intensity : float):
        return -log(random.random()) / intensity

    def __init__(self, intensity, service_process):
        if service_process not in rand_function.keys():
            raise ValueError("No such random number generator.")
        self.intensity = intensity
        self.service_time_generator = rand_function[service_process]
        self.service_type = service_process
        self.payload = 0

    
    def get_instensity(self):
        return self.intensity
    
    def get_service_time_generator(self):
        return self.service_time_generator
    
    def get_payload(self):
        return self.payload

    def set_intensity(self, int_):
        self.intensity = int_

    def set_service_time_generator(self, func : str):
        if service_process not in rand_function.keys():
            raise ValueError("No such random number generator.")
        self.service_time_generator = rand_function[func]
        self.service_type = func
    
    def set_payload(self, payload):
        """Set payload to the server. If the server is busy, then do not push payload to the server and return False
        
        Return
        ------
        True if payload was set succesfully and False if the server was busy.
        """
        if  isinstance(payload, int):
            if payload > 0:
                if not self.is_empty():
                    return False
                self.payload = payload
                return True
        raise TypeError("Payload can be only positive integer.")
    
    def is_empty(self):
        return not self.payload
    
    def service(self):
        """Return the end of the payload service time."""
        time = self.service_time_generator(self.intensity)
        return time
    
    def remove_payload(self):
        """Remove server current payload. Return True if payload removed, return False if the server was empty"""
        if self.is_empty():
            return False
        else:
            self.payload = 0
            return True
    
    def __str__(self):
        return "Server with service time:{}. Busy {}".format(self.service_type, not self.is_empty())
    
    def __repr__(self):
        return str(self)

    def __contains__(self, jobId_):
        return self.payload == jobId_




class Buffer():
    """Waiting space class

    Attributes
    ---------
    rule    [str]   Rule of buffer. Valid rules are FIFO.
    capacity    Positive int representing capacity of the buffer. Also possible to pass INF which represent infinite capacity
    size    [int]   Positive integer representing current size of the queue.
    
    """
    
    def __init__(self, rule_, capacity_ = INF):
        self.rule = rule_
        if rule_ == "FIFO":
            self.buffer = QueueDataStructure()
        else:
            raise ValueError("Unknown queue rule.")
        self.capacity = capacity_
        self.size = 0

    def __str__(self):
        return "Waiting space with rule {}, capacity {} and current size {}\nWaiting jobs:\n{}".format(self.rule, self.capacity, self.size, self.buffer)
    
    def __repr__(self):
        return str(self)
    
    def is_empty(self):
        return self.buffer.is_empty()
    
    def enqueue(self, jobId_):
        """sumary_line
        Return
        ------
        False if buffer is already full. True if job attached to the queue.
        """
        if self.capacity != INF:
            if self.capacity <= self.size:
                return False
        if not isinstance(jobId_, int):
            raise TypeError("Job id should be positive integer.")
        self.buffer.enqueue(jobId_)
        self.size += 1
        return True
    
    def dequeue(self):
        """Remove job from waiting space.
        
        Return
        ------
        JobId from waitnig space or False if waiting space was empty.
        """
        if self.buffer.is_empty():
            return False
        self.size -= 1
        return self.buffer.dequeue()


    def get_capacity(self):
        """Return capacity. If None, then cpacity is infinite."""
        return self.capacity
    
    def is_job_at_buffer(self, jobId = 0):
        """"Return True if the job with passed id is at buffer. If passed jobId 0, then return if any job is at buffer"""
        if jobId > 0:
            return self.buffer.find_element(jobId) != -1
        elif not jobId:
            return self.buffer.is_empty()
        else:
            raise ValueError("The is_job_at_buffer function is able to retreive only valid jobId or 0.")


        


class Queue():
    """
    Queue class

    Attributes
    ----------
    service_time_generator    [function]  function for generating time moments
    intensity   [float or Tuple of floats]  intensities for servers. If there is only on value provided and many servers then the same value is passed to all servers
    servers [list]  list which represent servers of the queue
    buffer  [Buffer]    buffer object representing waiting space. If queue have no buffor simpy initialize queue with rule_ = FIFO and buffer_cap_ = 0

    Methods
    -------
    enqueue_job

    process(event)
    process following eventg
    """

    
    def __init__(self, intensity, service_time, rule_ = "FIFO", buffer_cap_ = INF, servers_number : int = 1):
        #Input validation
        if isinstance(service_time, str):
            if isinstance(intensity, (float, int)):
                if servers_number > 0:
                    self.servers = [Server(intensity, service_time) for _ in range(servers_number)]
                else:
                    raise ValueError("Number of servers must be positive integer.")
            elif isinstance(intensity, tuple):
                if servers_number == len(intensity):
                    correct_data = [not isinstance(val, (float, int)) for val in intensity]
                    if  sum(correct_data):
                        raise TypeError("Intensity must be positive float or int")
                    self.servers = [Server(int_, service_time) for int_ in intensity]
                else:
                    raise ValueError("Number of servers and intensities must be the same")
            else:
                raise TypeError("Intensity must be float or tuple of floats or ints")
        elif isinstance(service_time, tuple):
            if len(service_time) != servers_number:
                raise ValueError("Number of servers and service times must be the same")
            for st in service_time:
                if st not in rand_function:
                    raise ValueError("No such random number generator.".format(st))
            if isinstance(intensity, (float, int)):
                if servers_number > 0:
                    self.servers = [Server(intensity, st) for st in service_time]
                else:
                    raise ValueError("Number of servers must be positive integer.")
            elif isinstance(intensity, tuple):
                if servers_number == len(intensity):
                    correct_data = [not isinstance(val, (float, int)) for val in intensity]
                    if  sum(correct_data):
                        raise TypeError("Intensity must be positive float or int")
                    self.servers = [Server(int_, service_time) for (int_, st) in zip(intensity, service_time)]
                else:
                    raise ValueError("Number of servers and intensities must be the same")
            else:
                raise TypeError("Intensity must be float or tuple of floats or ints")
        else:
            raise TypeError("Service time must be name of the supported random variable.")
        self.servers_number = servers_number
        self.buffer = Buffer(rule_, buffer_cap_)
        if buffer_cap_ != INF:
            self.total_capacity = buffer_cap_ + servers_number
        else:
            self.total_capacity = INF

    def __str__(self):
        servers = ""
        for server in self.servers:
            servers += str(server) + "\n"
        return "Queue with {} servers:\n{}\nBuffer:\n{}".format(self.servers_number, servers, self.buffer)
    
    def get_servers(self):
        return self.servers
    
    def get_buffer(self):
        return self.buffer
    
    def get_intensities(self):
        return [server.get_instensity() for server in self.servers]
    
    def get_payloads(self):
        return [server.get_payload() for server in self.servers]
    
    def set_intensities(self, ints_):
        """Set instensities to servers. If one value is passed, then all servers will set that intensity."""
        if len(ints_) == 1:
            ints_ = np.repeat(ints_, self.servers_number)
        elif len(ints_) != self.servers_number:
            raise ValueError("Intiensities number must correspond with servers number")
        for server, intensity in zip(self.servers, ints_):
            server.set_intensity(int_)
    
    def set_payload(self, payload, serverId = -1):
        """
        Set payload, to selected server. serverId is the index of the server on server list. Default serverId is -1, what is interpreted fo find empty server.
        
        Args
        ----
        payload [int]       Positive integer representing jobId
        serverId    [int]   Positive integer or -1 which is an index of the free server. If -1 passed then try to set payload to first founded empty server.
        Return
        ------
        Positive integer which represent serverId that took the jobId or -1 if payload was rejected by servers.
        """
        if serverId == -1:
            id_ = self.find_empty_server()
            if id_ == -1:
                return -1
            else:
                self.servers[id_].set_payload(payload)
                return id_
        status_ = self.servers[serverId].set_payload(payload)
        return status_ * serverId + (not status_) * -1
    
    def find_all_empty_servers(self):
        mask = [(id_, server.is_empty()) for id_, server in zip([*range(self.servers_number)], self.servers)]
        return mask
    
    def find_empty_server(self):
        for id_, server in zip([*range(self.servers_number)], self.servers):
            if server.is_empty():
                return id_
        return -1

    def get_servers_number(self):
        return self.servers_number

    def get_buffer_capacity(self):
        return self.buffer.get_capacity()
    
    def enqueue(self, jobId_):
        """Add job Id to the queue. 
        
        Return
        ------
        -2 if the job is missing, -1 if the job is placed in the buffer and serverId if it was placed at the server.
        """
        serverId = self.find_empty_server()
        if serverId == -1:
            if not self.buffer.enqueue(jobId_):
                return -2
            return -1
        else:
            self.servers[serverId].set_payload(jobId_)
            return serverId
    
    def is_job_at_buffer(self, jobId_):
        return self.buffer.is_job_at_buffer(jobId_)
    
    def is_job_at_server(self, jobId_, pos = False):
        if pos:
            try:
                return [server.payload for server in self.servers].index(jobId_)
            except:
                return -1
        return jobId_ in [server.payload for server in self.servers]
    
    def is_job_at_queue(self, jobId_):
        return self.is_job_at_buffer(jobId_) or self.is_job_at_server(jobId_)
    
    def process(self, jobId_):
        """Processing job arrival. It should be reminded that ARRIVAL event is the only one event that can be resolve here.
        Args
        ----
        jobId   [int]

        Return
        ------
        -2 if the job is missing,
        -1 if the job is waiting in the buffer
        positinve number representing finish of the service if the job is being serviced at the server.
        """
        result = self.enqueue(jobId_)
        if result in (-2, -1):
            return result
        else:
            return self.servers[result].service()
            

        


rand_function = {
    "Exp" : Server.rand_exp
}

class Event():
    event_type = [
        BIRTH,
        ARRIVAL,
        WAITING,
        MISSING,
        PROCESSING,
        LEAVING,
        CLOSED
    ]
    def __init__(self, jobId_, nodeId_, e_type,  e_time):
        if e_type not in Event.event_type:
            raise ValueError("Invalid event type.")
        
        if e_time < 0:
            raise ValueError("Invalid event occurence time.")

        self.jobId = jobId_
        self.nodeId = nodeId_
        self.type = e_type
        self.time = e_time
    
    def get_time(self):
        return self.time

    def get_type(self):
        return self.type
    
    def get_jobId(self):
        return self.jobId
    
    def get_nodeId(self):
        return self.nodeId
    
    def __str__(self):
        return "Event:{}\t for Job({})\t at time:{}".format(self.type, self.jobId, self.time)
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        if type(self) == type(other):
            return self.time == other.time
        elif type(other) in (type(1), type(.1)):
            return self.time == other
        elif type(other) == type(True):
            return self.type == other
        else:
            raise TypeError("Cannot compare with type {}".format(other))

    def __gt__(self, other) -> bool:
        if type(self) == type(other):
            return self.time > other.time
        elif type(other) in (type(1), type(.1)):
            return self.time > other
        else:
            raise TypeError("Cannot compare with type {}".format(other))

    def __lt__(self, other) -> bool:
        if type(self) == type(other):
            return self.time < other.time
        elif type(other) in (type(1), type(.1)):
            return self.time < other
        else:
            raise TypeError("Cannot compare with type {}".format(other))

    def __ge__(self,other) -> bool:
        return not self < other

    def __le__(self, other) -> bool:
        return not self > other
    
    def __add__(self, other) -> float:
        if type(self) == type(other):
            return self.time + other.time
        elif type(other) in (type(1), type(.1)):
            return self.time + other


class Job():
    """Job class
    
    Attributes
    ----------

    id  [int]   job id
    nodeId    [int]  actual nodeId
    birth_time  [int or float]  arrival to the whole system time
    status  [string]    actual status of the job. The valid statuses are BIRTH, ARRIVAL, WAITING


    Return: return_description
    """
    
    status = [
        BIRTH,
        ARRIVAL,
        WAITING,
        MISSING,
        PROCESSING,
        LEAVING,
        CLOSED
    ]

    def __init__(self, id_ : int, sys_arr_time, nodeId_):
        if sys_arr_time < 0:
            raise ValueError("Invalid event occurence time.")

        self.id = id_
        self.nodeId = nodeId_
        self.birth_time = sys_arr_time
        self.status = BIRTH
        self.history = [(sys_arr_time, BIRTH, nodeId_)]
        self.close_time = None
        self.total_time_in_system = 0

    def set_status(self, status, time_, nodeId_):
        if not status in Job.status:
            raise ValueError("Invalid status value")
        self.status = status
        self.history.append((time_, status, nodeId_))
        self.__compute_time_in_system__()
        self.nodeId = nodeId_
        if status == CLOSED:
            self.close_time = time_

    def __compute_time_in_system__(self):
        self.total_time_in_system = self.history[-1][0] - self.birth_time

    def get_id(self):
        return self.id

    def get_status(self):
        return self.status
    
    def get_close_time(self):
        return self.close_time
    
    def get_history(self):
        return self.history
    
    def get_total_time(self):
        return self.total_time_in_system
    
    def get_current_node(self):
        return self.nodeId

    def __str__(self):
        return "Job ({})\tSTATUS:{}\tNodeId:{}\nBirth:{}\tTotal time in system:{}\n".format(self.id, self.status, self.nodeId,  self.birth_time, self.total_time_in_system)
    
    def __repr__(self):
        return str(self)


class System():
    """sumary_line
    
    
    """
    
    def __init__(self, arrival_process_, use_name : bool = False, starting_name : str = "Arrival"):
        """sumary_line
        
        Arguments
        ---------
        arrival_process [tuple] tuple (process_type_string, intensity) or  number intensity for initialization an arrival process. If providen number then Poisson Process used.
        jobs    [list]  list of jobs in the system
        time    [float] Time value. It represent time from begining of system start. While the time is provided and system started then arrivals are generated and entering the system events added to event heap. During simulation components resolve jobs and genetate next event which are moved at the heap, therefore all events are consiedr i chronologic way.
        """
        if isinstance(arrival_process_, tuple):
            arrival_process = ArrivalProcess(arrival_process_[0], arrival_process_[1])
        else:
            arrival_process = ArrivalProcess(arrival_process_)
        self.system_graph = Graph(arrival_process, use_name, starting_name)
        self.jobs = {}
        self.event_heap = BinaryHeap()
        self.time = 0
        self.history = []
        self._toogle_names = use_name

    def add_node(self, prev_node, node, prob = 1, name = "") -> None:
        """Add new component to the system. It is crucial to point out that this method add route from existing node to this new node so you cannot pass Arrival Process.

        Agrs
        ----
        prev_node [string or int]   id if the already existing node in the graph or name of the this node if use_name is set to True.
        node [Queue or None]   Object for the node. Queue if it is QueueVertex or None if it is LeavingVertex.
        prob    [float] probability/weight for the route. Default is 1.
        name    [str]   name of the vertex. If use_name was set to True for the graph it will be use to represent a vertex.
        Return
        ------
        None
        """
        self.system_graph.add_vert(node, prev_node, prob, name)

    def add_route(self, prev_node, next_node, prob = 1):
        """Add route beetwen two components of the system.
        
        Agrs
        ----
        prev_node [string or int]   id if the already existing node in the graph or name of the this node if use_name is set to True.
        next_node [string or int]   id if the already existing node in the graph or name of the this node if use_name is set to True.
        Return
        ------
        None
        """
        self.system_graph.add_route(prev_node, next_node, prob)
    
    def get_queues(self):
        """Return QueueVerices"""
        return [vertex for vertex in self.system_graph.get_vertices().values() if type(vertex) == QueueVertex]
    
    def get_arrival_processes(self):
        return [vertex for vertex in self.system_graph.get_vertices().values() if type(vertex) == ArrivalVertex]

    def get_vertices(self):
        return self.system_graph.get_vertices()

    def run(self, time_, verbose = False):
        if time_ < 0:
            raise ValueError("Finish time should be positive number.")
        # Generate events 
        num = 0
        events = []
        for arrVert in self.get_arrival_processes():
            arrVert._set_arrival_counter_(num)
            jobs, ev = arrVert.generate_arrivals(time_)
            arrVert.generate_arrivals(time_)
            jobsIds = [*map(lambda t: t.get_id(), jobs)]
            self.jobs.update({id_ : job for id_, job in zip(jobsIds, jobs)})
            events.extend(ev)
            num = arrVert.get_arrivals_number()
        self.event_heap.build_heap(events)
        # Trigger all events
        while not self.event_heap.is_empty():
            ev = self.event_heap.pop()
            if verbose:
                print(ev)
            # Store all parameters expected to trigger next events
            ev_time = ev.get_time()
            ev_job = self.jobs[ev.get_jobId()]
            ev_node = self.system_graph.get_vertex(ev.get_nodeId())
            self.history.append(ev)
            if ev.get_type() == BIRTH:
                events = ev_node.dequeue(ev.get_jobId(), ev_time)
                for ev in events:
                    self.event_heap.insert(ev)
                # map(lambda t: self.event_heap.insert(t), events)
            elif ev.get_type() in (ARRIVAL, MISSING, CLOSED, WAITING, PROCESSING):
                continue
                #ARRIVAL event is triggered during dequeue, but dequeue immediately trigger enqueue method of the next node so all actions are performed
                #MISSING event is triggered in enqueue to QueueVertex and immediately set job status to missing. Nothing to do.
                #CLOSED event is triggered in enqueue to LeavingVertex. Nothing to do. 
            elif ev.get_type() == LEAVING:
                events = ev_node.dequeue(ev.get_jobId(), ev_time)
                for ev in events:
                    self.event_heap.insert(ev)
                # map(lambda t: self.event_heap.insert(t), events)

    def get_jobs(self) -> list:
        return list(self.jobs.values())

    def get_history(self) -> list:
        return self.history

    def __str__(self):
        return "{}".format(self.system_graph)
    
    def __repr__(self):
        return str(self)
    
    def reset(self):
        self.jobs = []
        self.event_heap = BinaryHeap()
        self.time = 0
        self.history = []
        map(lambda t: t.reset(), self.get_arrival_processes())

class Timer():
    """
    Class holding event heap and time
    
    Attributes
    ----------
    _time [number]              represent current time
    _event_heap [BinaryHeap]    event heap

    """
    
    def __init__(self, link : Graph = None, seed = None):
        self.link = link
        self._time = 0
        if seed:
            self._generator = RandomState(MT19937(SeedSequence(seed)))
        else:
            self._generator = None
        self._event_heap = BinaryHeap()
        self._history = []
        
    
    # def move_to(self, time_):
    #     if not self.link:
    #         raise TypeError("Cannot run simulation if the graph is not specified for the timer.")
    #     if isinstance(time_, (int, float)):
    #         if time_ >= 0:
    #             while time_ >= self._event_heap.get_min():
    #                 # Here we perform all actions
    #                 # Get the event check the status and perform approrpiate action
    #                 e = self._event_heap.pop()
    #                 e_time = e.get_time()
    #                 # if e

    #             return
    #         raise ValueError("Time should be non-negative number.")        
            
    #     else:
    #         raise TypeError("Time should be number.") 
        
    def push_event(self, event_):
        if not isinstance(event_, Event):
            raise TypeError("Event heap store only event object, but {} passed".format(type(event_)))
        self._event_heap.insert(event_)
        self._history.append(event_)

    def pop_event(self) -> Event:
        """"Pick event from heap and move to it's time moement."""
        
        e = self._event_heap.pop()
        self._time = e.get_time()
        return e
    
    def get_time(self):
        return self._time
    
    def get_history(self):
        return self._history
    



if __name__ == "__main__":
    #Test heap
    heap = BinaryHeap()
    heap.build_heap(list(np.random.randint(40, size = 10)))
    print(heap)
    while not heap.is_empty():
        print(heap.pop())

    #Test event class
    ev = Event(0, BIRTH, 3)
    print(ev)
    print(ev.get_time())
    print(ev.get_type())

    #Test arrival process
    print(ArrivalProcess.poisson_process(10,.3))
    arrPr = ArrivalProcess(3)
    arrPr.generate_arrivals(4)
    print(arrPr.get_arrivals())

    #Test System
    system = System(3)
    
    print(system)

    #Test queues
    print("Test queues...")
    qu = Queue(3, "Exp", "FIFO", 4)
    print(qu)

    qu = Queue(3, "Exp")
    print(qu)

    #Test job
    print("Testing job object...")
    jj = Job(2, 0, 11)
    jj.set_status(ARRIVAL, 11, 1)
    jj.set_status(WAITING, 11, 1)
    jj.set_status(PROCESSING, 13, 1)
    print(jj)
    jj.set_status(LEAVING, 17, 1)
    jj.set_status(CLOSED, 17, 2)
    print(jj)
    print(jj.get_close_time())
    print(jj.get_history())
