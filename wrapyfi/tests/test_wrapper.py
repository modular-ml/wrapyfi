import unittest
import importlib
import multiprocessing
from multiprocessing import Queue
import queue
import time


class ZeroMQTestPublisher(unittest.TestCase):
    MWARE = "zeromq"

    def test_two_node_close(self):
        import wrapyfi.tests.tools.class_test as class_test
        importlib.reload(class_test)
        Test = class_test.Test
        if self.MWARE not in Test.get_communicators():
            self.skipTest(f"{self.MWARE} not installed")

        # create two node nodes to test closing and auto-naming of the second node
        test = Test()
        test2 = Test()
        test.activate_communication(test.exchange_object, mode="publish")
        test2.activate_communication(test2.exchange_object, mode="publish")
        # ensure both nodes are registered
        self.assertIn("Test.exchange_object", Test._MiddlewareCommunicator__registry)
        self.assertIn("Test.exchange_object.2", Test._MiddlewareCommunicator__registry)

        for i in range(2):
            msg_object, = test.exchange_object(mware=self.MWARE)
            msg_object2, = test2.exchange_object(mware=self.MWARE, topic="/test/test_native_exchange2")
            self.assertDictEqual(msg_object, msg_object2)
        test.close()
        del test

        # ensure only one node is registered
        self.assertIn("Test.exchange_object", Test._MiddlewareCommunicator__registry)
        self.assertNotIn("Test.exchange_object.2", Test._MiddlewareCommunicator__registry)
        for i in range(2):
            msg_object2, = test2.exchange_object(mware=self.MWARE, topic="/test/test_native_exchange2")
        test2.close()
        del test2

        # ensure no nodes are registered
        self.assertNotIn("Test.exchange_object", Test._MiddlewareCommunicator__registry)

    def test_three_node_close(self):
        import wrapyfi.tests.tools.class_test as class_test
        importlib.reload(class_test)
        Test = class_test.Test
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
        self.assertNotIn("Test.exchange_object", Test._MiddlewareCommunicator__registry)

    def test_publish_listen_same_node(self):
        import wrapyfi
        import wrapyfi.tests.tools.class_test as class_test
        importlib.reload(wrapyfi)
        importlib.reload(class_test)
        test_func = class_test.test_func

        if self.MWARE not in class_test.Test.get_communicators():
            self.skipTest(f"{self.MWARE} not installed")

        listen_queue = Queue(maxsize=10)
        test_lsn = multiprocessing.Process(target=test_func, args=(listen_queue,),
                                    kwargs={"mode": "listen", "mware": self.MWARE, "iterations": 10,
                                            "should_wait": True})
        # test_lsn.daemon = True

        publish_queue = Queue(maxsize=10)
        test_pub = multiprocessing.Process(target=test_func, args=(publish_queue,),
                                    kwargs={"mode": "publish", "mware": self.MWARE, "iterations": 10,
                                            "should_wait": True})
        # test_pub.daemon = True

        test_lsn.start()
        test_pub.start()
        test_lsn.join()
        test_pub.join()
        for i in range(10):
            try:
                self.assertDictEqual(listen_queue.get(timeout=3), publish_queue.get(timeout=3))
            except queue.Empty:
                self.assertEqual(i, 9)


class ROS2TestPublisher(ZeroMQTestPublisher):
    MWARE = "ros2"


class YarpTestPublisher(ZeroMQTestPublisher):
    MWARE = "yarp"


class ROSTestPublisher(ZeroMQTestPublisher):
    MWARE = "ros"


if __name__ == '__main__':
    unittest.main()
