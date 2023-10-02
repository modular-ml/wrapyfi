import abc
import io
import json
import base64
from datetime import datetime

import numpy as np

from wrapyfi.utils import *


class JsonEncoder(json.JSONEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs.get('serializer_kwargs', {}))
        self.plugins = dict()
        for plugin_key, plugin_val in PluginRegistrar.encoder_registry.items():
            self.plugins[plugin_key] = plugin_val(**kwargs)

    def find_plugin(self, obj):
        for cls in reversed(type(obj).__mro__[:-1]):
            if cls.__module__ == 'collections.abc':
                continue  # skip classes from collections.abc
            if issubclass(cls, abc.ABCMeta):
                if cls.__abstractmethods__:
                    continue  # skip abstract classes with abstract methods
            return self.plugins.get(cls, None)
        return None

    def encode(self, obj):
        def hint_tuples(item):
            if isinstance(item, tuple):
                return dict(__wrapyfi__=('tuple', item))
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {key: hint_tuples(value) for key, value in item.items()}
            else:
                return item

        return super(JsonEncoder, self).encode(hint_tuples(obj))

    def default(self, obj):

        if isinstance(obj, set):
            return dict(__wrapyfi__=('set', list(obj)))

        elif isinstance(obj, datetime):
            return dict(__wrapyfi__=('datetime', obj.isoformat()))

        elif isinstance(obj, np.datetime64):
            return dict(__wrapyfi__=('numpy.datetime64', str(obj)))

        elif isinstance(obj, (np.ndarray, np.generic)):
            with io.BytesIO() as memfile:
                np.save(memfile, obj)
                obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
            return dict(__wrapyfi__=('numpy.ndarray', obj_data))

        plugin_match = self.find_plugin(obj)
        if plugin_match is not None:
            detected, plugin_return = plugin_match.encode(obj)
            if detected:
                return plugin_return

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class JsonDecodeHook(object):
    def __init__(self, **kwargs):
        self.plugins = dict()
        for plugin_key, plugin_val in PluginRegistrar.decoder_registry.items():
            self.plugins[plugin_key] = plugin_val(**kwargs)

    def object_hook(self, obj):

        if isinstance(obj, dict):
            wrapyfi = obj.get('__wrapyfi__', None)
            if wrapyfi is not None:
                obj_type = wrapyfi[0]

                if obj_type == 'tuple':
                    return tuple(wrapyfi[1])

                elif obj_type == 'set':
                    return set(wrapyfi[1])

                elif obj_type == 'datetime':
                    return datetime.fromisoformat(wrapyfi[1])

                elif obj_type == 'numpy.datetime64':
                    return np.datetime64(wrapyfi[1])

                elif obj_type == 'numpy.ndarray':
                    with io.BytesIO(base64.b64decode(wrapyfi[1].encode('ascii'))) as memfile:
                        return np.load(memfile)

                plugin_match = self.plugins.get(obj_type, None)
                if plugin_match is not None:
                    detected, plugin_return = plugin_match.decode(obj_type, wrapyfi)
                    if detected:
                        return plugin_return

        return obj
