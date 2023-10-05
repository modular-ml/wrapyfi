import argparse
import logging
logging.getLogger().setLevel(logging.INFO)

import zmq

from wrapyfi.middlewares.zeromq import ZeroMQMiddlewareReqRep, ZeroMQMiddlewarePubSub


def main(comm_type, socket_ip, socket_pub_port, socket_sub_port, socket_rep_port, socket_req_port, **kwargs):
    if comm_type == "pubsub":
        socket_pub_address = f"tcp://{socket_ip}:{socket_pub_port}"
        socket_sub_address = f"tcp://{socket_ip}:{socket_sub_port}"

        ZeroMQMiddlewarePubSub.activate(**{"socket_pub_address": socket_pub_address,
                                           "socket_sub_address": socket_sub_address,
                                           "proxy_broker_spawn": "process", "verbose": True}, **kwargs)
        while True:
            pass
    elif comm_type == "reqrep":
        socket_rep_address = f"tcp://{socket_ip}:{socket_rep_port}"
        socket_req_address = f"tcp://{socket_ip}:{socket_req_port}"
        ZeroMQMiddlewareReqRep.activate(**{"socket_rep_address": socket_rep_address,
                                           "socket_req_address": socket_req_address,
                                           "proxy_broker_spawn": "process", "verbose": True}, **kwargs)
        while True:
            pass

    elif comm_type == "pubsubpoll":  # debugging the xpub/xsub with a poller
        xpub_addr = f"tcp://{socket_ip}:{socket_pub_port}"
        xsub_addr = f"tcp://{socket_ip}:{socket_sub_port}"
        context = zmq.Context()
        xpub_socket = context.socket(zmq.XPUB)
        xpub_socket.bind(xpub_addr)
        xsub_socket = context.socket(zmq.XSUB)
        xsub_socket.bind(xsub_addr)

        poller = zmq.Poller()
        poller.register(xpub_socket, zmq.POLLIN)
        poller.register(xsub_socket, zmq.POLLIN)
        logging.info(f"[ZeroMQ] Intialising PUB/SUB device broker")
        while True:
            # get event
            event = dict(poller.poll(1000))
            if xpub_socket in event:
                message = xpub_socket.recv_multipart()
                #print("[ZeroMQ BROKER] xpub_socket recv message: %r" % message)
                xsub_socket.send_multipart(message)
            if xsub_socket in event:
                message = xsub_socket.recv_multipart()
                #print("[ZeroMQ BROKER] xsub_socket recv message: %r" % message)
                xpub_socket.send_multipart(message)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket_ip", type=str, default="127.0.0.1", help="Socket IP address")
    parser.add_argument("--socket_pub_port", type=int, default=5555, help="Socket publishing port")
    parser.add_argument("--socket_sub_port", type=int, default=5556, help="Socket subscription port")
    parser.add_argument("--socket_rep_port", type=int, default=5559, help="Socket reply port")
    parser.add_argument("--socket_req_port", type=int, default=5560, help="Socket request port")
    parser.add_argument("--comm_type", type=str, default="pubsub", choices=["pubsub", "pubsubpoll", "reqrep"],
                        help="The zeromq communication pattern")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))


