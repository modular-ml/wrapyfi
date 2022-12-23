import abc
import io
import json
import base64

import numpy as np

from wrapyfi.utils import *


class JsonEncoder(json.JSONEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs.get('serializer_kwargs', {}))
        self.plugins = dict()
        for plugin_key, plugin_val in PluginRegistrar.encoder_registry.items():
            self.plugins[plugin_key] = plugin_val(**kwargs)

    def default(self, obj):

        if isinstance(obj, tuple):
            return dict(__wrapyfi__=('tuple', obj))

        elif isinstance(obj, set):
            return dict(__wrapyfi__=('set', list(obj)))

        elif isinstance(obj, (np.ndarray, np.generic)):
            with io.BytesIO() as memfile:
                np.save(memfile, obj)
                obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
            return dict(__wrapyfi__=('numpy.ndarray', obj_data))
        meta_type = type(obj).__mro__[-2]
        plugin_match = self.plugins.get(meta_type if meta_type is not abc.ABC else type(obj).__mro__[-3], None)
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

                elif obj_type == 'numpy.ndarray':
                    with io.BytesIO(base64.b64decode(wrapyfi[1].encode('ascii'))) as memfile:
                        return np.load(memfile)

                plugin_match = self.plugins.get(obj_type, None)
                if plugin_match is not None:
                    detected, plugin_return = plugin_match.decode(obj_type, wrapyfi)
                    if detected:
                        return plugin_return

        return obj
