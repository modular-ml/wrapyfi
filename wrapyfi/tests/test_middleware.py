import unittest
import multiprocessing
from multiprocessing import Queue
import queue


class ZeroMQTestMiddleware(unittest.TestCase):
    MWARE = "zeromq"

    def test_publish_listen(self):
        import wrapyfi.tests.tools.class_test as class_test
        # importlib.reload(wrapyfi)
        # importlib.reload(class_test)
        test_func = class_test.test_func
        Test = class_test.Test
        Test.close_all_instances()

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


class ROS2TestMiddleware(ZeroMQTestMiddleware):
    MWARE = "ros2"


class YarpTestMiddleware(ZeroMQTestMiddleware):
    MWARE = "yarp"


class ROSTestMiddleware(ZeroMQTestMiddleware):
    MWARE = "ros"


if __name__ == '__main__':
    unittest.main()
