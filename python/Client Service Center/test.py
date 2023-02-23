import unittest
from simulation import *
import math

class TestSimulation(unittest.TestCase):
    """
    Test
    """
    def test_lambda_limit(self):
        simul = Simulation(lambda x: math.sin(x), 20, 10)
        maxim = simul.get_lambda()
        msg = "f(x) = sin(x)\t should be aproximately 1. Got {}".format(maxim)
        self.assertAlmostEqual(simul.get_lambda(), 1, 3, msg)

        simul = Simulation(lambda x: math.atan(x), 20, 10)
        maxim = simul.get_lambda()
        msg = "f(x) = arc tan(x)\t should be aproximately pi/2. Got {}".format(maxim)
        self.assertAlmostEqual(simul.get_lambda(), math.pi/2, 3, msg)

        simul = Simulation(lambda x: (7-x)*(x-1)*(x+2)*(x+2), 20, 10)
        maxim = simul.get_lambda()
        msg = "f(x) = sin(x)\t should be aproximately 1. Got {}".format(maxim)
        self.assertAlmostEqual(simul.get_lambda(), 392.694, 3, msg)

        simul = Simulation(lambda x: 3, 20, 10)
        maxim = simul.get_lambda()
        msg = "f(x) = 3\t should be aproximately 3. Got {}".format(maxim)
        self.assertAlmostEqual(simul.get_lambda(), 3, 3, msg)


class TestEvent(unittest.TestCase):
    def test_logical_operators(self):
        event1 = Event(CALL, 32)
        #self.assertEqual(event1, CALL, "Event is call so event1 == CALL should be True")
        self.assertGreater(event1, 12, "Event time is 32 so 32>12 should be True")
        self.assertGreater(event1, 11.34, "Event time is 32 so 32>11.34 should be True")
        self.assertLess(event1, 45, "Event time is 32 so 32<45 should be True")
        self.assertLess(event1, 45.23, "Event time is 32 so 32<45.23 should be True")
        self.assertEqual(event1, 32, "Event time is 32 so 32=32 should be True")
        self.assertEqual(event1, 32.0, "Event time is 32 so 32=32.0 should be True")
        self.assertGreaterEqual(event1, 32.0, "Event time is 32 so 32>=32.0 should be True")
        self.assertGreaterEqual(event1, 32, "Event time is 32 so 32>=32 should be True")
        self.assertGreaterEqual(event1, 10, "Event time is 32 so 32>=10 should be True")
        self.assertGreaterEqual(event1, 10.0, "Event time is 32 so 32>=10.0 should be True")
        self.assertLessEqual(event1, 32.0, "Event time is 32 so 32<=32.0 should be True")
        self.assertLessEqual(event1, 32, "Event time is 32 so 32<=32 should be True")
        self.assertLessEqual(event1, 33, "Event time is 32 so 32<=33 should be True")
        self.assertLessEqual(event1, 33.0, "Event time is 32 so 32<=33.0 should be True")

        event2 = Event(END_OF_RING, 12)
        self.assertGreater(event1, event2, "Event times are 32 and 12 as follow so 32 > 12 should be True")

    def test_add_operator(self):
        event = Event(CALL, 3)
        self.assertEqual(event + 4, 7, "Event(CALL,3) + 4 should be 7")
        self.assertEqual(event + 2.5, 5.5, "Event(CALL,3) + 2.5 should be 5.5")
        event2 = Event(END_OF_RING,1.2)
        self.assertEqual(event + event2, 4.2, "Event(CALL,3)+Event(END_OF_RING,1.2) should be 4.2")

if __name__ == "__main__":
    unittest.main()