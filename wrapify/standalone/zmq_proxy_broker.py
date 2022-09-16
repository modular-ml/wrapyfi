import argparse

import zmq

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket_ip", type=str, default="127.0.0.1", help="The socket ip address")
    parser.add_argument("--socket_port", type=str, default="5555", help="The socket publishing port")
    parser.add_argument("--socket_sub_port", type=str, default="5556", help="The socket subscription port")
    parser.add_argument("--carrier", type=str, default="tcp", help="The communication protocol")
    return parser.parse_args()

args = parse_args()
xpub_addr = f"{args.carrier}://{args.socket_ip}:{args.socket_port}"
xsub_addr = f"{args.carrier}://{args.socket_ip}:{args.socket_sub_port}"
context = zmq.Context()

#create XPUB
xpub_socket = context.socket(zmq.XPUB)
xpub_socket.bind(xpub_addr)

#create XSUB
xsub_socket = context.socket(zmq.XSUB)
xsub_socket.bind(xsub_addr)

#create poller
poller = zmq.Poller()
poller.register(xpub_socket, zmq.POLLIN)
poller.register(xsub_socket, zmq.POLLIN)

while True:

    # get event
    event = dict(poller.poll(1000))
    if xpub_socket in event:
        message = xpub_socket.recv_multipart()
        print("[ZeroMQ BROKER] xpub_socket recv message: %r" % message)
        xsub_socket.send_multipart(message)
    if xsub_socket in event:
        message = xsub_socket.recv_multipart()
        print("[ZeroMQ BROKER] xsub_socket recv message: %r" % message)
        xpub_socket.send_multipart(message)