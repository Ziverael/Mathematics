import unittest
import logging
import queues as qq
import numpy as np


TESTCASE = "GLOBAL"

class ContextFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.
    """
    def filter(self, record):
        record.TESTCASE = TESTCASE
        return True

class ArrivalProcess(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    f_handler = logging.FileHandler("{}-ArrivalProcess.log".format(__name__))
    formatter = logging.Formatter("%(name)s::%(TESTCASE)s::%(levelname)s::%(message)s")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    
    
    def test_seed(self):
        global TESTCASE
        TESTCASE = "test_seed"
        self.logger.addFilter(ContextFilter())
        self.logger.info("Create two Arrival Processes with the same seed.")
        arr = qq.ArrivalProcess(2, seed_ = 0)
        t = 5
        self.logger.info("Create arrivals for arr t={}.".format(t))
        arr.generate_arrivals(5)
        arrs = arr.get_arrivals()
        self.logger.debug(arrs)
        arr2 = qq.ArrivalProcess(2, seed_ = 0)
        self.logger.info("Create arrivals for arr2 t={}.".format(t))
        arr2.generate_arrivals(5)
        arrs2 = arr2.get_arrivals()
        self.logger.debug(arrs2)
        self.assertEqual(list(arrs), list(arrs2), msg = "Arrivals are different for arr and arr2")
        
        
        
        self.logger.info("Create External RNG and replace value.")
        self.logger.info("Pass RNG to two identical Arrival Processes.")
        #Two get two identical lists Generator should reset it seed after every simulation
        rng = np.random.default_rng(seed = 5)
        arr1 = qq.ArrivalProcess(2)
        arr1.set_generator(rng)
        arr1.generate_arrivals(t)

        rng = np.random.default_rng(seed = 5)
        arr2 = qq.ArrivalProcess(2)
        arr2.set_generator(rng)
        arr2.generate_arrivals(t)
        arrs1 = arr1.get_arrivals()
        arrs2 = arr2.get_arrivals()
        self.logger.debug("Arrivals for arr1")
        self.logger.debug(arrs1)
        self.logger.debug("Arrivals for arr2")
        self.logger.debug(arrs2)
        self.assertEqual(list(arrs1), list(arrs2), msg = "Expected equal lists")
        self.logger.debug("Arrivals for both arrival're lists are identical.")



        self.logger.info("Repeat generating arrivals with the same seed.")
        arr = qq.ArrivalProcess(2, seed_ = 0)
        t = 5
        self.logger.info("Create arrivals for arr t={}.".format(t))
        arr.generate_arrivals(5)
        arrs = arr.get_arrivals()
        self.logger.debug(arrs)
        self.logger.info("Create arrivals for arr t={}.".format(t))
        arr.set_seed(0)
        arr.reset()
        arr.generate_arrivals(5)
        arrs2 = arr.get_arrivals()
        self.logger.debug(arrs2)
        self.assertEqual(list(arrs), list(arrs2), msg = "Arrivals are different for arr and arr2")



        # #It is invalid operation. It not worked as expected
        # self.logger.info("Replace seed for rng and check if simulation will be identical.")
        # rng = np.random.default_rng(seed = 5)
        # arr1.reset()
        # arr1.generate_arrivals(t)
        # arrs3 = arr1.get_arrivals()
        # self.logger.debug("Arrivals for arr3")
        # self.logger.debug(arrs3)
        # self.assertEqual(list(arrs1), list(arrs3), msg = "Expected equal lists")


        self.logger.info("Create two identical generators.")
        rng1 = np.random.default_rng(seed = 5)
        rng2 = np.random.default_rng(seed = 5)
        arr1 = qq.ArrivalProcess(2)
        arr2 = qq.ArrivalProcess(2)
        arr1.set_generator(rng1)
        arr2.set_generator(rng2)
        arr1.generate_arrivals(t)
        arr2.generate_arrivals(t)
        arrs1 = arr1.get_arrivals()
        arrs2 = arr2.get_arrivals()
        self.logger.debug("Arrivals for arr1 t={}".format(t))
        self.logger.debug(arrs1)
        self.logger.debug("Arrivals for arr2 t={}".format(t))
        self.logger.debug(arrs2)
        self.assertEqual(list(arrs1), list(arrs2), msg = "Expected equal lists")


class ArrivalVertex(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    f_handler = logging.FileHandler("{}-ArrivalVertex.log".format(__name__))
    formatter = logging.Formatter("%(name)s::%(TESTCASE)s::%(levelname)s::%(message)s")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    
    
    def test_seed(self):
        global TESTCASE
        TESTCASE = "test_seed"
        self.logger.addFilter(ContextFilter())
        
        
        self.logger.info("Create two Arrival ProcesseVertices with the same seed.")
        arr = qq.ArrivalVertex(1, qq.ArrivalProcess(2, seed_ = 0))
        # arr.set_seed(0)
        t = 5
        self.logger.info("Create arrivals for arr t={}.".format(t))
        arr.generate_arrivals(t)
        arrs = arr.get_history()["Events"][0]
        arrsStr = "\n".join([ str(ev) for ev in arr.get_history()["Events"][0]])
        self.logger.debug(arrsStr)
        arr2 = qq.ArrivalVertex(2, qq.ArrivalProcess(2, seed_ = 0))
        # arr2.set_seed(0)
        self.logger.info("Create arrivals for arr2 t={}.".format(t))
        arr2.generate_arrivals(5)
        arrs2 = arr2.get_history()["Events"][0]
        arrs2Str = "\n".join([ str(ev) for ev in arr2.get_history()["Events"][0]])
        self.logger.debug(arrs2Str)
        for ev in arrs:
            self.assertIn(ev, arrs2, msg = "{} not in arr2".format(ev))
        


        self.logger.info("Repeat generating arrivals with the same seed.")
        arr.set_seed(0)
        t = 5
        self.logger.info("Create arrivals for arr t={}.".format(t))
        arr.generate_arrivals(t)
        arrs3 = arr.get_history()["Events"][0]
        arrs3Str = "\n".join([ str(ev) for ev in arr.get_history()["Events"][0]])
        self.logger.debug(arrs3Str)
        for ev in arrs3:
            self.assertIn(ev, arrs, msg = "{} not in arr2".format(ev))



        
        
        # self.logger.info("Create External RNG and replace value.")
        # self.logger.info("Pass RNG to two identical Arrival Processes.")
        # #Two get two identical lists Generator should reset it seed after every simulation
        # rng = np.random.default_rng(seed = 5)
        # arr1 = qq.ArrivalProcess(2, gen_ = rng)
        # arr1.generate_arrivals(t)

        # rng = np.random.default_rng(seed = 5)
        # arr2 = qq.ArrivalProcess(2, gen_ = rng)
        # arr2.generate_arrivals(t)
        # arrs1 = arr1.get_arrivals()
        # arrs2 = arr2.get_arrivals()
        # self.logger.debug("Arrivals for arr1")
        # self.logger.debug(arrs1)
        # self.logger.debug("Arrivals for arr2")
        # self.logger.debug(arrs2)
        # self.assertEqual(list(arrs1), list(arrs2), msg = "Expected equal lists")
        # self.logger.debug("Arrivals for both arrival're lists are identical.")



        # #It is invalid operation. It not worked as expected
        # self.logger.info("Replace seed for rng and check if simulation will be identical.")
        # rng = np.random.default_rng(seed = 5)
        # arr1.reset()
        # arr1.generate_arrivals(t)
        # arrs3 = arr1.get_arrivals()
        # self.logger.debug("Arrivals for arr3")
        # self.logger.debug(arrs3)
        # self.assertEqual(list(arrs1), list(arrs3), msg = "Expected equal lists")


        self.logger.info("Create two identical generators.")
        rng1 = np.random.default_rng(seed = 5)
        rng2 = np.random.default_rng(seed = 5)
        arr1 = qq.ArrivalProcess(2)
        arr1.set_generator(rng1)
        arr2 = qq.ArrivalProcess(2)
        arr2.set_generator(rng2)
        arr1.generate_arrivals(t)
        arr2.generate_arrivals(t)
        arrs1 = arr1.get_arrivals()
        arrs2 = arr2.get_arrivals()
        self.logger.debug("Arrivals for arr1 t={}".format(t))
        self.logger.debug(arrs1)
        self.logger.debug("Arrivals for arr2 t={}".format(t))
        self.logger.debug(arrs2)
        self.assertEqual(list(arrs1), list(arrs2), msg = "Expected equal lists")




class Queue(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    f_handler = logging.FileHandler("{}-Queue.log".format(__name__))
    formatter = logging.Formatter("%(name)s::%(TESTCASE)s::%(levelname)s::%(message)s")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    
    
    def test_seed(self):
        global TESTCASE
        TESTCASE = "test_seed"
        self.logger.addFilter(ContextFilter())
        
        
        self.logger.info("Create two Queues with the same seed.")
        qu1 = qq.Queue(1, "Exp", seed_ = 0)
        qu2 = qq.Queue(1, "Exp", seed_ = 0)
        n = 10
        self.logger.info("Generate service times for qu1.")
        times1 = [qu1.servers[0].service() for i in range(n)]
        self.logger.debug(times1)
        self.logger.info("Generate service times for qu2.")
        times2 = [qu2.servers[0].service() for i in range(n)]
        self.logger.debug(times2)
        for t in times1:
            self.assertIn(t, times2, msg = "{} not in times2".format(t))
        for t in times2:
            self.assertIn(t, times1, msg = "{} not in times1".format(t))
        


        self.logger.info("Repeat generating arrivals with the same seed.")
        qu1.set_seed(0)
        qu1.reset()
        self.logger.info("Generate service times for qu1.")
        times2 = [qu1.servers[0].service() for i in range(n)]
        self.logger.debug(times1)
        for t in times1:
            self.assertIn(t, times2, msg = "{} not in times2".format(t))
        for t in times2:
            self.assertIn(t, times1, msg = "{} not in times1".format(t))
        

class ArrivalVertex(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    f_handler = logging.FileHandler("{}-ProcessVertex.log".format(__name__))
    formatter = logging.Formatter("%(name)s::%(TESTCASE)s::%(levelname)s::%(message)s")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    
    
    def test_seed(self):
        global TESTCASE
        TESTCASE = "test_seed"
        self.logger.addFilter(ContextFilter())
        
        
        self.logger.info("Create two Arrival QueueVertices with the same seed.")
        v1 = qq.QueueVertex(1, qq.Queue(1, "Exp"), seed_ = 0)
        v2 = qq.QueueVertex(1, qq.Queue(1, "Exp"), seed_ = 0)
        t = 5
        # self.logger.info("Create jobs for arr t={}.".format(t))
        # arr.generate_arrivals(t)
        # arrs = arr.get_history()["Events"][0]
        # arrsStr = "\n".join([ str(ev) for ev in arr.get_history()["Events"][0]])
        # self.logger.debug(arrsStr)
        # arr2 = qq.ArrivalVertex(2, qq.ArrivalProcess(2, gen_ = 0))
        # # arr2.set_seed(0)
        # self.logger.info("Create arrivals for arr2 t={}.".format(t))
        # arr2.generate_arrivals(5)
        # arrs2 = arr2.get_history()["Events"][0]
        # arrs2Str = "\n".join([ str(ev) for ev in arr2.get_history()["Events"][0]])
        # self.logger.debug(arrs2Str)
        # for ev in arrs:
        #     self.assertIn(ev, arrs2, msg = "{} not in arr2".format(ev))
        


        # self.logger.info("Repeat generating arrivals with the same seed.")
        # arr.set_seed(0)
        # t = 5
        # self.logger.info("Create arrivals for arr t={}.".format(t))
        # arr.generate_arrivals(t)
        # arrs3 = arr.get_history()["Events"][0]
        # arrs3Str = "\n".join([ str(ev) for ev in arr.get_history()["Events"][0]])
        # self.logger.debug(arrs3Str)
        # for ev in arrs3:
        #     self.assertIn(ev, arrs, msg = "{} not in arr2".format(ev))
