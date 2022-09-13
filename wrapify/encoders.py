import io
import json
import base64

import numpy as np

from wrapify.utils import *


class JsonEncoder(json.JSONEncoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plugins = dict()
        for plugin_key, plugin_val in PluginRegistrar.registry.items():
            self.plugins[plugin_key] = plugin_val()

    def default(self, obj):

        if isinstance(obj, set):
            return dict(__wrapify__=('set', list(obj)))

        elif isinstance(obj, np.ndarray):
            with io.BytesIO() as memfile:
                np.save(memfile, obj)
                obj_data = base64.b64encode(memfile.getvalue()).decode('ascii')
            return dict(__wrapify__=('numpy.ndarray', obj_data))

        for plugin in self.plugins.values():
            detected, plugin_return = plugin.encode(obj)
            if detected:
                return plugin_return

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class JsonDecodeHook(object):

    def __init__(self, **kwargs):
        self.plugins = dict()
        for plugin_key, plugin_val in PluginRegistrar.registry.items():
            self.plugins[plugin_key] = plugin_val(**kwargs)

    def object_hook(self, obj):

        if isinstance(obj, dict):
            wrapify = obj.get('__wrapify__', None)
            if wrapify is not None:
                obj_type = wrapify[0]

                if obj_type == 'set':
                    return set(wrapify[1])

                elif obj_type == 'numpy.ndarray':
                    with io.BytesIO(base64.b64decode(wrapify[1].encode('ascii'))) as memfile:
                        return np.load(memfile)

                for plugin in self.plugins.values():
                    detected, plugin_return = plugin.decode(obj_type, wrapify)
                    if detected:
                        return plugin_return

        return obj
