import logging

from wrapyfi.connect.listeners import Listener, Listeners, ListenerWatchDog


@Listeners.register("MMO", "fallback")
class FallbackListener(Listener):

    def __init__(self, name: str, in_topic: str, carrier: str = "tcp",
                 should_wait: bool = True, missing_middleware_object: str = "", **kwargs):
        logging.warning(f"Fallback listener employed due to missing middleware or object type: "
                        f"{missing_middleware_object}")
        Listener.__init__(self, name, in_topic, carrier=carrier, should_wait=should_wait, **kwargs)
        self.missing_middleware_object = missing_middleware_object

    def establish(self, repeats=-1, **kwargs):
        return None

    def listen(self):
        return None

    def close(self):
        return None