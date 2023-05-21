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

class QueuingSystem(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    f_handler = logging.FileHandler("{}-QueuingSystem.log".format(__name__))
    formatter = logging.Formatter("%(name)s::%(TESTCASE)s::%(levelname)s::%(message)s")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    
    
    def test_build_system(self):
        global TESTCASE
        TESTCASE = "test_build_system"
        self.logger.addFilter(ContextFilter())
        self.logger.info("Building simple System")
        qsys = qq.System(2, True, "Arr")
        qsys.add_node("Arr", qq.Queue(2, "Exp"), name = "Q1")
        qsys.add_node("Q1", None, name = "END")
        self.logger.info("Simple queue initialize")
        self.logger.debug(qsys)
    
    def test_system_vertices_validity(self):
        global TESTCASE
        TESTCASE = "test_system_vertices_validity"
        self.logger.addFilter(ContextFilter())
        

        self.logger.info("Building System with two identical vetex names")
        qsys = qq.System(2, True, "Arr")
        qsys.add_node("Arr", qq.Queue(2, "Exp"), name = "Q1")
        with self.assertRaises(ValueError) as cm:
            qsys.add_node("Arr", qq.Queue(4, "Exp"), name = "Q1")
        qsys.add_node("Q1", None, name = "END")
        self.assertEqual(str(cm.exception), "Given node alerady exists in queuing system.")
        self.logger.debug("Raises correct exception.")


        self.logger.info("Building System withou LeavingVertex object.")
        qsys = qq.System(1, True, "Arr")
        qsys.add_node("Arr", qq.Queue(1, "Exp"), name = "Q1")
        self.logger.info("Queuing system initialize.")
        self.logger.debug(qsys)
        self.logger.info("Trying qsys.run(1)")
        with self.assertRaises(qq.QueueingSystemError) as cm:
            qsys.run(1)
        self.logger.debug("Raises message: {}".format(str(cm.exception)))
        if str(cm.exception) == "System do not has LeavingVertex object.":
            self.logger.debug("Raises correct exception.")
        else:
            self.logger.error("Raises incorrect exception.")
        self.assertEqual(str(cm.exception), "System do not has LeavingVertex object.")
        
        self.logger.info("Trying qsys.initialize_simulation(1)")
        with self.assertRaises(qq.QueueingSystemError) as cm:
            qsys.initialize_simulation(1)
        self.logger.debug("Raises message: {}".format(str(cm.exception)))
        if str(cm.exception) == "System do not has LeavingVertex object.":
            self.logger.debug("Raises correct exception.")
        else:
            self.logger.error("Raises incorrect exception.")
        self.assertEqual(str(cm.exception), "System do not has LeavingVertex object.")
        
        self.logger.info("Trying qsys.move_to(1)")
        with self.assertRaises(qq.QueueingSystemError) as cm:
            qsys.move_to(1)
        self.logger.debug("Raises message: {}".format(str(cm.exception)))
        if str(cm.exception) == "System do not has LeavingVertex object.":
            self.logger.debug("Raises correct exception.")
        else:
            self.logger.error("Raises incorrect exception.")
        self.assertEqual(str(cm.exception), "System do not has LeavingVertex object.")




        self.logger.info("Building System with QueueingVertex without access to LeavingVertex.")
        qsys = qq.System(1, True, "Arr")
        qsys.add_node("Arr", qq.Queue(1, "Exp"), prob = 0.3, name = "Q1")
        qsys.add_node("Arr", qq.Queue(2, "Exp"), prob = 0.6, name = "Q2")
        qsys.add_node("Arr", None, prob = 0.1, name = "END")
        self.logger.info("Queuing system initialize")
        self.logger.debug(qsys)
        self.logger.info("Trying qsys.run(1)")
        with self.assertRaises(qq.QueueingSystemError) as cm:
            qsys.run(1)
        self.logger.debug("Raises message: {}".format(str(cm.exception)))
        if str(cm.exception) == "QueuingVertex Q1 do not have access to LeavingVertex.":
            self.logger.debug("Raises correct exception.")
        else:
            self.logger.error("Raises incorrect exception.")
        self.assertEqual(str(cm.exception), "QueuingVertex Q1 do not have access to LeavingVertex.")
        
        self.logger.info("Trying qsys.initialize_simulation(1)")
        with self.assertRaises(qq.QueueingSystemError) as cm:
            qsys.initialize_simulation(1)
        self.logger.debug("Raises message: {}".format(str(cm.exception)))
        if str(cm.exception) == "QueuingVertex Q1 do not have access to LeavingVertex.":
            self.logger.debug("Raises correct exception.")
        else:
            self.logger.error("Raises incorrect exception.")
        self.assertEqual(str(cm.exception), "QueuingVertex Q1 do not have access to LeavingVertex.")
        
        self.logger.info("Trying qsys.move_to(1)")
        with self.assertRaises(qq.QueueingSystemError) as cm:
            qsys.move_to(1)
        self.logger.debug("Raises message: {}".format(str(cm.exception)))
        if str(cm.exception) == "QueuingVertex Q1 do not have access to LeavingVertex.":
            self.logger.debug("Raises correct exception.")
        else:
            self.logger.error("Raises incorrect exception.")
        self.assertEqual(str(cm.exception), "QueuingVertex Q1 do not have access to LeavingVertex.")
        

        
        self.logger.info("Building System with Few LeavingVertex objects")
        qsys = qq.System(1, True, "Arr")
        qsys.add_node("Arr", qq.Queue(1, "Exp"), prob = 0.4, name = "Q1")
        qsys.add_node("Q1", None, prob = 1, name = "END")
        with self.assertRaises(TypeError) as cm:
            qsys.add_node("Arr", None, prob = 0.6, name = "END2")
        self.logger.debug("Raises message: {}".format(str(cm.exception)))
        self.logger.debug("Raises message: {}".format(str(cm.exception)))
        if str(cm.exception) == "QueuingVertex Q1 do not have access to LeavingVertex.":
            self.logger.debug("Raises correct exception.")
        else:
            self.logger.debug("Raises correct exception.")
        self.assertEqual(str(cm.exception), "In queuing system must be only one LeavingVertex.")
        
    


    def test_system_seed(self):
        global TESTCASE
        TESTCASE = "test_system_seed"
        self.logger.addFilter(ContextFilter())
        
        
        self.logger.info("Verified if all objects gets the same seed.")
        self.logger.info("Building System with fixed seed.")
        qsys = qq.System(1, True, "Arr", seed_ = 0)
        qsys.add_node("Arr", qq.Queue(0.5, "Exp", servers_number = 3, buffer_cap_ = 5), name = "Q1")
        qsys.add_node("Q1", None, name = "Endpoint")
        self.logger.info("System initialized.")
        self.logger.debug(qsys)
        seeds = ""
        for vert in qsys.get_vertices().values():
            seeds += "\nVertex no.{} seed:{}".format(vert.get_key(), vert.get_seed())
            if type(vert) == qq.ArrivalVertex:
                seeds += "\nArrival Process seed:{}".format(vert.obj.get_seed())
            elif type(vert) == qq.QueueVertex:
                seeds += "\nQueue seed:{}".format(vert.obj.get_seed())
                for server in vert.get_servers():
                    seeds += "\nServer seed:{}".format(server.get_seed())
        self.logger.debug(seeds)
        for vert  in qsys.get_vertices().values():
            self.assertEqual(0, vert.get_seed(), msg = "Vertex {} have seed {}".format(vert.get_key(), vert.get_seed()))
        
        
        
        
        
        
        params = {
            "lambda" : .5,
            "mu1" : .3,
            "mu2" : .5,
            "mu3" : .1
        }
        # self.logger.info("Building System with fiex seed.")
        # qsys = qq.System(params["lambda"], True, "Arr", seed_ = 150)
        # qsys.add_node("Arr", qq.Queue(params["mu1"], "Exp", servers_number = 3, buffer_cap_ = 5), name = "Q1")

        # qsys.add_node("Q1", qq.Queue(params["mu1"], "Exp", servers_number = 1, buffer_cap_ = qq.INF), prob = .3, name = "Q2")
        # qsys.add_node("Q1", qq.Queue(params["mu2"], "Exp", servers_number = 1, buffer_cap_ = qq.INF), prob = .3, name = "Q3")
        # qsys.add_node("Q1", qq.Queue(params["mu3"], "Exp", servers_number = 1, buffer_cap_ = qq.INF), prob = .2, name = "Q4")

        # qsys.add_node("Q1", None, prob = .2, name = "Endpoint")
        # qsys.add_route("Q2", "Endpoint")
        # qsys.add_route("Q3", "Endpoint")
        # qsys.add_route("Q4", "Endpoint")
        self.logger.info("Building System with fiex seed.")
        qsys = qq.System(params["lambda"], True, "Arr", seed_ = 0)
        qsys.add_node("Arr", qq.Queue(params["mu1"], "Exp", servers_number = 3, buffer_cap_ = 5), name = "Q1")
        qsys.add_node("Q1", None, name = "Endpoint")
        self.logger.info("System initialized.")
        self.logger.debug(qsys)
        self.logger.info("Run qsys.run(10)  two times and compare histories.")
        self.logger.info("Run qsys.run(10)")
        qsys.run(10)
        hist1 = qsys.get_history()
        hist1_rep = "\n".join([str(ev) for ev in qsys.get_history()])
        self.logger.debug("History for simulation no.1")
        self.logger.debug(hist1_rep)
        self.logger.info("Reset system and seed.")
        qsys.set_seed(0)
        qsys.reset()
        qsys.run(10)
        hist2 = qsys.get_history()
        hist2_rep = "\n".join([str(ev) for ev in qsys.get_history()])
        self.logger.debug("History for simulation no.2")
        self.logger.debug(hist2_rep)
        for ev in hist1:
            self.assertIn(ev, hist2, "Event {} not appeared in 2nd simulation.".format(ev))
        for ev in hist2:
            self.assertIn(ev, hist1, "Event {} not appeared in 1nd simulation.".format(ev))


        self.logger.info("Run qsys.move_to(5)  two times and compare histories.")
        qsys.reset()
        qsys.set_seed(150)
        self.logger.info("Run qsys.initialize_simulation(10)")
        qsys.initialize_simulation(10)
        self.logger.info("Run qsys.move_to(5)")
        qsys.move_to(5)
        hist2 = qsys.get_history()
        hist2_rep = "\n".join([str(ev) for ev in qsys.get_history()])
        self.logger.debug("History for simulation no.1")
        self.logger.debug(hist2_rep)
        self.logger.info("Reset system and seed.")
        qsys.set_seed(150)
        qsys.reset()
        qsys.initialize_simulation(10)
        qsys.move_to(5)
        hist3 = qsys.get_history()
        hist3_rep = "\n".join([str(ev) for ev in qsys.get_history()])
        self.logger.debug("History for simulation no.2")
        self.logger.debug(hist3_rep)
        for ev in hist2:
            self.assertIn(ev, hist1, "Event {} not appeared in run(10) history.".format(ev))
        for ev in hist2:
            self.assertIn(ev, hist3, "Event {} not appeared in 1nd simulation.".format(ev))
        for ev in hist3:
            self.assertIn(ev, hist2, "Event {} not appeared in 2nd simulation.".format(ev))


    def test_move_to(self):
        global TESTCASE
        TESTCASE = "test_move_to"
        self.logger.addFilter(ContextFilter())
        
        self.logger.info("Building simple System")
        qsys = qq.System(1, True, "Arr")
        qsys.add_node("Arr", qq.Queue(2, "Exp"), name = "Q1")
        qsys.add_node("Q1", None, name = "END")
        self.logger.info("Simple queue initialize")
        self.logger.debug(qsys)
        self.logger.info("Compare if move_to(7) return the same results as move_to(5) and move_to(2, relative = True)")
        T = 10
        t = 7
        self.logger.info("Initialize system for T={} and move to t={}".format(T, t))
        qsys.initialize_simulation(T)
        qsys.move_to(t)
        hist1 = qsys.get_history()
        hist1_rep = ""
        for ev in hist1:
            hist1_rep +="\n{}".format(ev)
        self.logger.debug("History for simulation no.1")
        self.logger.debug(hist1_rep)
        qsys.reset(reset_seed = True)
        qsys.initialize_simulation(T)
        print(qsys.total_time, qsys.time)
        qsys.move_to(5)
        qsys.move_to(2, relative = True)
        self.logger.debug("History for simulation no.2")
        hist2 = qsys.get_history()
        hist2_rep = ""
        for ev in hist2:
            hist2_rep +="\n{}".format(ev)
        self.logger.debug(hist2_rep)
        for ev in hist2:
            self.assertIn(ev, hist1, msg = "{} not in simulation 1.".format(ev))
        for ev in hist1:
            self.assertIn(ev, hist2, msg = "{} not in simulation 2.".format(ev))
        

        t1 = 7
        t2 = 5
        self.logger.info("Move to t1={} and go backwards to t2={}".format(t1, t2))
        qsys.set_seed(0)
        qsys.reset()
        qsys.initialize_simulation(t1)
        qsys.move_to(t1)
        hist1_rep = "\n".join([str(ev) for ev in qsys.get_history()])
        hist1 = qsys.get_history()
        self.logger.debug("History for move_to(t1)")
        self.logger.debug(hist1_rep)
        #Manually
        qsys.set_seed(0)
        qsys.reset()
        qsys.initialize_simulation(t1)
        qsys.move_to(t2)
        #It should works
        # qsys.move_to(t2)
        hist2_rep = "\n".join([str(ev) for ev in qsys.get_history()])
        self.logger.debug("History for move_to(t2)")
        self.logger.debug(hist2_rep)
        hist2 = qsys.get_history()
        # Check if all events have time to t2
        self.logger.debug("Verify if all events in hist2 is not greater than {} and appeared in hist1.".format(t2))
        for ev in hist2:
            self.assertIn(ev, hist1, msg = "{} not in hist1.".format(ev))
            self.assertTrue(ev <= t2, msg = "{} appeared after {}.".format(ev, t2))






        # qv = [vert._generator for vert in qsys.get_vertices().values() if type(vert) == qq.QueueVertex]
        # print(qv)
        # print(qsys._generator)
