try:
    from importlib import metadata
    __version__ = metadata.version(__name__)
    __url__ = metadata.metadata(__name__)["Home-page"]
except ImportError:
    import pkg_resources
    __version__ = pkg_resources.require("wrapyfi")[0].version
    __url__ = "https://github.com/fabawi/wrapyfi/"

from wrapyfi.utils import PluginRegistrar

PluginRegistrar.scan()

import logging
logging.getLogger().setLevel(logging.INFO)
