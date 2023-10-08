import logging

from wrapyfi.connect.servers import Server, Servers


@Servers.register("MMO", "fallback")
class FallbackServer(Server):

    def __init__(self, name: str, out_topic: str, carrier: str = "", missing_middleware_object: str = "", **kwargs):
        logging.warning(f"Fallback server employed due to missing middleware or object type: "
                        f"{missing_middleware_object}")
        Server.__init__(self, name, out_topic, carrier=carrier, **kwargs)
        self.missing_middleware_object = missing_middleware_object

    def establish(self, repeats=-1, **kwargs):
        return None

    def await_request(self, *args, **kwargs):
        return args, kwargs

    def reply(self, obj):
        return obj

    def close(self):
        return None
