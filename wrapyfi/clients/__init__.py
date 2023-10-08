import logging

from wrapyfi.connect.clients import Client, Clients


@Clients.register("MMO", "fallback")
class FallbackClient(Client):

    def __init__(self, name: str, in_topic: str, carrier: str = "", missing_middleware_object: str = "", **kwargs):
        logging.warning(f"Fallback client employed due to missing middleware or object type: "
                        f"{missing_middleware_object}")
        Client.__init__(self, name, in_topic, carrier=carrier, **kwargs)
        self.missing_middleware_object = missing_middleware_object

    def establish(self, repeats=-1, **kwargs):
        return None

    def request(self, *args, **kwargs):
        return None

    def _request(self, *args, **kwargs):
        return None

    def _await_reply(self):
        return None

    def close(self):
        return None