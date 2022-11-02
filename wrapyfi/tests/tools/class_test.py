import numpy as np
import time

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class Test(MiddlewareCommunicator):

    @MiddlewareCommunicator.register("NativeObject", "$mware", "Test", "$topic",
                                     should_wait="$should_wait")
    def exchange_object(self, msg=None, mware=DEFAULT_COMMUNICATOR, topic="/test/test_native_exchange", should_wait=False):
        ret = {"message": msg,
               "set": {'a', 1, None},
               "list": [[[3, [4], 5.677890, 1.2]]],
               "string": "string of characters",
               "2": 2.73211,
               "dict": {"other": [None, False, 16, 4.32,]}}
        return ret,


def test_func(queue_buffer, mode="listen", mware=DEFAULT_COMMUNICATOR, topic="/test/test_native_exchange", iterations=2,
              should_wait=False):
    test = Test()
    test.activate_communication(test.exchange_object, mode=mode)
    for i in range(iterations):
        my_message = test.exchange_object(msg=f"signal_idx:{i}", mware=mware, topic=topic, should_wait=should_wait)
        if my_message is not None:
            print(f"result {mode}:", my_message[0]["message"])
            queue_buffer.put(my_message[0])
        time.sleep(0.5)