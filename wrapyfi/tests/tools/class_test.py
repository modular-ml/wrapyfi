import numpy as np

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR


class Test(MiddlewareCommunicator):

    @MiddlewareCommunicator.register("NativeObject", "$mware", "Test", "$topic",
                                     should_wait=False)
    def exchange_object(self, mware=DEFAULT_COMMUNICATOR, topic="/test/test_native_exchange"):
        ret = {"set": {'a', 1, None},
               "list": [[[3, [4], 5.677890, 1.2]]],
               "string": "string of characters",
               2: 2.73211,
               "dict": {"other": (None, False, np.int(16), 4.32,)}}
        return ret,
