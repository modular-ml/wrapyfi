import logging
import threading
import multiprocessing
import time

import zmq


update_trigger = False
param_server = None
root_topics = set()
cached_params = {}
params = {"WRAPYFI_ACTIVE": "True"}


def __init_broker():
    global update_trigger

    xpub_addr = "tcp://*:5555"
    xsub_addr = "tcp://*:5655"
    context = zmq.Context()
    # create XPUB
    xpub_socket = context.socket(zmq.XPUB)
    xpub_socket.bind(xpub_addr)
    # create XSUB
    xsub_socket = context.socket(zmq.XSUB)
    xsub_socket.bind(xsub_addr)

    # create poller
    poller = zmq.Poller()
    poller.register(xpub_socket, zmq.POLLIN)
    poller.register(xsub_socket, zmq.POLLIN)
    logging.info(f"[ZeroMQ] Intialising PUB/SUB device broker")
    while True:
        # get event
        event = dict(poller.poll(1))
        if xpub_socket in event:
            message = xpub_socket.recv_multipart()
            print("[ZeroMQ BROKER] xpub_socket recv message: %r" % message)
            if message[0].startswith(b"\x00"):
                root_topics.remove(message[0][1:].decode("utf-8"))
            elif message[0].startswith(b"\x01"):
                root_topics.add(message[0][1:].decode("utf-8"))
            xsub_socket.send_multipart(message)

        if xsub_socket in event:
            message = xsub_socket.recv_multipart()
            print("[ZeroMQ BROKER] xsub_socket recv message: %r" % message)
            if message[0].startswith(b"\x01") or message[0].startswith(b"\x00"):
                xpub_socket.send_multipart(message)
            else:
                fltr_key = message[0].decode("utf-8")
                fltr_message = {key: val for key, val in params.items()
                                if key.startswith(fltr_key)}
                print("[ZeroMQ BROKER] xsub_socket filtered message: %r" % fltr_message)
                for key, val in fltr_message.items():
                    prefix, param = key.rsplit("/", 1) if "/" in key else ("", key)
                    xpub_socket.send_multipart([prefix.encode("utf-8"), param.encode("utf-8"), val.encode("utf-8")])
            # xpub_socket.send_multipart(message)

        if event:
            update_trigger = True

        print(event)
        if param_server is not None:
            publish_params(param_server)


def publish_params(param_server):
    global update_trigger, cached_params

    # Check if there are any subscribed clients before publishing updates
    if not root_topics:
        return

    if not any((update_trigger, root_topics)) and cached_params == params:
        return
    else:
        time.sleep(0.01)
        update_trigger = False
        cached_params = params.copy()

    # Publish updates for all parameters to subscribed clients
    for key, val in params.items():
        prefix, param = key.rsplit("/", 1) if "/" in key else ("", key)
        param_server.send_multipart([prefix.encode("utf-8"), param.encode("utf-8"), val.encode("utf-8")])


def request_handler(request_server):
    while True:
        # Receive requests from clients and handle them accordingly
        request = request_server.recv_string()
        if request.startswith("get"):
            try:
                # Extract the parameter name and namespace prefix from the request
                prefix, param = request[4:].rsplit("/", 1) if "/" in request[4:] else ("", request[4:])
                # Construct the full parameter name with the namespace prefix
                full_param = "/".join([prefix, param]) if prefix else param
                if full_param in params:
                    request_server.send_string(str(params[full_param]))
                else:
                    # Send an error message if the parameter does not exist
                    request_server.send_string("error:::parameter does not exist")
            except ValueError:
                # Send an error message if the request is malformed
                request_server.send_string("error:::malformed request")
        elif request.startswith("read"):
            try:
                # Extract the parameter name and namespace prefix from the request
                prefix = request[5:]
                # Construct the full parameter name with the namespace prefix
                if any(param.startswith(prefix) for param in params.keys()):
                    request_server.send_string(f"success:::{prefix}")
                else:
                    # Send an error message if the parameter does not exist
                    request_server.send_string("error:::parameter does not exist")
            except ValueError:
                # Send an error message if the request is malformed
                request_server.send_string("error:::malformed request")
        elif request.startswith("set"):
            try:
                # Extract the parameter name, namespace prefix and value from the request
                prefix, param, value = request[4:].rsplit("/", 2)
                # Construct the full parameter name with the namespace prefix
                full_param = "/".join([prefix, param]) if prefix else param
                params[full_param] = value
                request_server.send_string(f"success:::{prefix}")
            except ValueError:
                # Send an error message if the request is malformed
                request_server.send_string("error:::malformed request")
        elif request.startswith("delete"):
            try:
                # Extract the parameter name and namespace prefix from the request
                prefix, param = request[7:].rsplit("/", 1)
                # Construct the full parameter name with the namespace prefix
                full_param = "/".join([prefix, param]) if prefix else param
                if full_param in params:
                    del params[full_param]
                    request_server.send_string(f"success:::{prefix}")
                else:
                    # Send an error message if the parameter does not exist
                    request_server.send_string("error:::parameter does not exist")
            except ValueError:
                # Send an error message if the request is malformed
                request_server.send_string("error:::malformed request")
        else:
            request_server.send_string("error:::invalid request")


def main(broker_spawn="process"):
    if broker_spawn == "multiprocessing":
        proxy = multiprocessing.Process(name='zeromq_broker', target=__init_broker, kwargs={})
        proxy.daemon = True
        proxy.start()
    else:  # if threaded
        proxy = threading.Thread(name='zeromq_broker', target=__init_broker, kwargs={})
        proxy.setDaemon(True)
        proxy.start()

    # Create a ZeroMQ context
    context = zmq.Context()
    global param_server
    # Bind a PUB socket to the parameter server endpoint
    param_server = context.socket(zmq.PUB)
    param_server.connect("tcp://127.0.0.1:5655")

    # Bind a REP socket to the request endpoint
    request_server = context.socket(zmq.REP)
    request_server.bind("tcp://*:5556")

    request_handler(request_server)
    pass


if __name__ == "__main__":
    main()
