import argparse
import paddle

from wrapyfi.connect.wrapper import MiddlewareCommunicator, DEFAULT_COMMUNICATOR

"""
A message publisher and listener for Paddlepaddle tensors

Here we demonstrate
1. Using the NativeObject message
2. Transmit a nested dummy python object with native objects and multidim torch tensors
3. Demonstrating the flipping of devices by mapping CPU to GPU and vice versa

Run:
    # On machine 1 (or process 1): Publisher waits for keyboard and transmits message
    python3 paddle_tensor.py --mode publish
    # On machine 2 (or process 2): Listener waits for message and prints the entire dummy object
    python3 paddle_tensor.py --mode listen

"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="publish", choices={"publish", "listen"}, help="The transmission mode")
    parser.add_argument("--mware", type=str, default=DEFAULT_COMMUNICATOR, choices=MiddlewareCommunicator.get_communicators(),
                        help="The middleware to use for transmission")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    class Notify(MiddlewareCommunicator):

        @MiddlewareCommunicator.register("NativeObject", args.mware, "Notify", "/notify/test_native_exchange",
                                         carrier="", should_wait=True,
                                         # load_torch_device='cuda:0', map_torch_devices={'cpu': 'cuda:0', 'cuda:0': 'cpu'})
                                         listener_kwargs=dict(load_paddle_device='gpu:0',
                                                              map_paddle_devices={'cpu': 'cuda:0', 'gpu:0': 'cpu'}))

        def exchange_object(self):
            msg = input("Type your message: ")
            ret = {"message": msg,
                   "paddle_ones": paddle.Tensor(paddle.ones((2, 4)), place=paddle.CPUPlace()),
                   "paddle_zeros_cuda": paddle.Tensor(paddle.zeros((2, 3)), place=paddle.CUDAPlace(0))}
            return ret,

    notify = Notify()

    notify.activate_communication(Notify.exchange_object, mode=args.mode)
    while True:
        msg_object, = notify.exchange_object()
        print("Method result:", msg_object)
