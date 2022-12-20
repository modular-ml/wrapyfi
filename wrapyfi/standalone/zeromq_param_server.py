
from wrapyfi.middlewares.zeromq import ZeroMQMiddlewareParamServer


def main(**kwargs):
    ZeroMQMiddlewareParamServer.activate(**{"socket_address": "tcp://127.0.0.1:5655",
                             "socket_sub_address": "tcp://127.0.0.1:5656",
                             "server_address": "tcp://127.0.0.1:5659",
                             "proxy_broker_spawn": "process"}, **kwargs)
    while True:
        pass


if __name__ == "__main__":
    main()
