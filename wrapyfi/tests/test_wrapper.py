import unittest
import importlib
import multiprocessing
from multiprocessing import Queue
import queue
import time


class ZeroMQTestWrapper(unittest.TestCase):
    MWARE = "zeromq"

    def test_activate_communication(self):
        import wrapyfi.tests.tools.class_test as class_test
        Test = class_test.Test
        Test.close_all_instances()
        if self.MWARE not in Test.get_communicators():
            self.skipTest(f"{self.MWARE} not installed")

        # create three node nodes to test closing and auto-naming of the second node
        test, test2, test3 = Test(), Test(), Test()
        test.activate_communication(test.exchange_object, mode="publish")
        test2.activate_communication(test2.exchange_object, mode=None)
        test2.activate_communication(test2.exchange_object, mode="publish")  # override None
        test3.activate_communication(test3.exchange_object, mode="disable")

        # ensure both nodes are registered
        self.assertIn("Test.exchange_object", Test._MiddlewareCommunicator__registry)
        self.assertIn("Test.exchange_object.2", Test._MiddlewareCommunicator__registry)
        self.assertIn("Test.exchange_object.3", Test._MiddlewareCommunicator__registry)
        for i in range(2):
            msg_object, = test.exchange_object(mware=self.MWARE)
            msg_object2, = test2.exchange_object(mware=self.MWARE, topic="/test/test_native_exchange2")
            msg_object3, = test3.exchange_object(mware=self.MWARE)
            self.assertDictEqual(msg_object, msg_object2)
            self.assertIsNone(msg_object3)
        test.close()
        del test

        # ensure only one node is registered
        self.assertIn("Test.exchange_object", Test._MiddlewareCommunicator__registry)
        self.assertIn("Test.exchange_object.2", Test._MiddlewareCommunicator__registry)
        self.assertNotIn("Test.exchange_object.3", Test._MiddlewareCommunicator__registry)

        for i in range(2):
            msg_object2, = test2.exchange_object(mware=self.MWARE, topic="/test/test_native_exchange2")
        test2.close()
        test3.close()
        del test2, test3

        # ensure no nodes are registered
        self.assertIn("Test.exchange_object", Test._MiddlewareCommunicator__registry)

    def test_close(self):
        import wrapyfi.tests.tools.class_test as class_test
        Test = class_test.Test
        Test.close_all_instances()

        if self.MWARE not in Test.get_communicators():
            self.skipTest(f"{self.MWARE} not installed")

        # create three nodes to test reordering on closing
        test, test2, test3 = Test(), Test(), Test()
        test.activate_communication(test.exchange_object, mode="disable")
        test2.activate_communication(test2.exchange_object, mode=None)
        test3.activate_communication(test3.exchange_object, mode="publish")

        # ensure all nodes are registered
        self.assertIn("Test.exchange_object", Test._MiddlewareCommunicator__registry)
        self.assertIn("Test.exchange_object.2", Test._MiddlewareCommunicator__registry)
        self.assertIn("Test.exchange_object.3", Test._MiddlewareCommunicator__registry)

        # delete the second node. The third node should be renamed to second
        test2.close()
        del test2
        self.assertIn("Test.exchange_object", Test._MiddlewareCommunicator__registry)
        self.assertIn("Test.exchange_object.2", Test._MiddlewareCommunicator__registry)
        self.assertNotIn("Test.exchange_object.3", Test._MiddlewareCommunicator__registry)

        # check that the first & third node are still registered
        test_id = Test._MiddlewareCommunicator__registry["Test.exchange_object"]["__WRAPYFI_INSTANCES"].index(hex(id(test)))
        test3_id = Test._MiddlewareCommunicator__registry["Test.exchange_object"]["__WRAPYFI_INSTANCES"].index(hex(id(test3)))
        self.assertEqual(test_id, 0)
        self.assertEqual(test3_id, 1)

        # close the two nodes
        test.close()
        del test

        # recreate test node which should be renamed to second, and the third node should be renamed to first (no index)
        test = Test()
        test.activate_communication(test.exchange_object, mode="publish")
        self.assertIn("Test.exchange_object", Test._MiddlewareCommunicator__registry)

        # check that the first & third node are still registered
        test_id = Test._MiddlewareCommunicator__registry["Test.exchange_object"]["__WRAPYFI_INSTANCES"].index(hex(id(test)))
        test3_id = Test._MiddlewareCommunicator__registry["Test.exchange_object"]["__WRAPYFI_INSTANCES"].index(hex(id(test3)))
        self.assertEqual(test_id, 1)
        self.assertEqual(test3_id, 0)

        # ensure no nodes are registered
        test3.close()
        del test3
        test.close()
        del test

        self.assertIn("Test.exchange_object", Test._MiddlewareCommunicator__registry)
        self.assertNotIn("Test.exchange_object.2", Test._MiddlewareCommunicator__registry)
        self.assertNotIn("Test.exchange_object.3", Test._MiddlewareCommunicator__registry)

        # close all instances after all have been closed
        Test.close_all_instances()
        self.assertIn("Test.exchange_object", Test._MiddlewareCommunicator__registry)
        self.assertNotIn("Test.exchange_object.2", Test._MiddlewareCommunicator__registry)
        self.assertNotIn("Test.exchange_object.3", Test._MiddlewareCommunicator__registry)

    def test_get_communicators(self):
        import wrapyfi.tests.tools.class_test as class_test
        # importlib.reload(class_test)
        Test = class_test.Test

        if self.MWARE not in Test.get_communicators():
            self.skipTest(f"{self.MWARE} not installed")

        self.assertIn(self.MWARE, Test.get_communicators())


class ROS2TestWrapper(ZeroMQTestWrapper):
    MWARE = "ros2"


class YarpTestWrapper(ZeroMQTestWrapper):
    MWARE = "yarp"


class ROSTestWrapper(ZeroMQTestWrapper):
    MWARE = "ros"


if __name__ == '__main__':
    unittest.main()