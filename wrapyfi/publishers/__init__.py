import logging

from wrapyfi.connect.publishers import Publisher, Publishers


@Publishers.register("MMO", "fallback")
class FallbackPublisher(Publisher):

    def __init__(self, name: str, out_topic: str, carrier: str = "",
                 should_wait: bool = True, missing_middleware_object: str = "", **kwargs):
        logging.warning(f"Fallback publisher employed due to missing middleware or object type: "
                        f"{missing_middleware_object}")
        Publisher.__init__(self, name, out_topic, carrier=carrier, should_wait=should_wait, **kwargs)
        self.missing_middleware_object = missing_middleware_object

    def establish(self, repeats=-1, **kwargs):
        return None

    def publish(self, obj):
        return obj

    def close(self):
        return None