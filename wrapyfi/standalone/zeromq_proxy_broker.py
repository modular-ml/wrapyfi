import argparse
import logging
logging.getLogger().setLevel(logging.INFO)

import zmq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket_ip", type=str, default="127.0.0.1", help="The socket ip address")
    parser.add_argument("--socket_pub_port", type=str, default="5555", help="The socket publishing port")
    parser.add_argument("--socket_sub_port", type=str, default="5556", help="The socket subscription port")
    parser.add_argument("--socket_rep_port", type=str, default="5559", help="The socket request port")
    parser.add_argument("--socket_req_port", type=str, default="5560", help="The socket reply port")
    parser.add_argument("--carrier", type=str, default="tcp", help="The communication protocol")
    parser.add_argument("--type", type=str, default="pubsub", choices=["pubsub", "pubsubpoll", "repreq"], help="The zeromq communication pattern")
    return parser.parse_args()

args = parse_args()

if args.type == "pubsub":
    in_port = args.socket_pub_port
    out_port = args.socket_sub_port
    in_socket_type = zmq.XPUB
    out_socket_type = zmq.XSUB
    comm_type = "PUB/SUB"
elif args.type == "repreq":
    in_port = args.socket_rep_port
    out_port = args.socket_req_port
    in_socket_type = zmq.XREP
    out_socket_type = zmq.XREQ
    comm_type = "REP/REQ"

if args.type == "pubsubpoll":  # debugging the xpub/xsub with a poller
    xpub_addr = f"{args.carrier}://{args.socket_ip}:{args.socket_pub_port}"
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
    logging.info(f"[ZeroMQ] Intialising PUB/SUB device broker")
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
else:
    context = zmq.Context()
    frontend = context.socket(in_socket_type)
    frontend.bind(f"{args.carrier}://{args.socket_ip}:{in_port}")
    backend = context.socket(out_socket_type)
    backend.bind(f"{args.carrier}://{args.socket_ip}:{out_port}")
    logging.info(f"[ZeroMQ] Intialising {comm_type} device broker")
    zmq.proxy(frontend, backend)
