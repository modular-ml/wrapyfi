__version__ = "0.4.14"
__url__ = "https://github.com/fabawi/wrapyfi/"
name = "wrapyfi"

try:
    from importlib import metadata
    __version__ = metadata.version(__name__)
    __url__ = metadata.metadata(__name__)["Home-page"]
except ImportError:
    try:
        import pkg_resources
        __version__ = pkg_resources.require(name)[0].version
    except pkg_resources.DistributionNotFound:
        pass
except Exception:
    pass

from wrapyfi.utils import PluginRegistrar

PluginRegistrar.scan()

import logging
logging.getLogger().setLevel(logging.INFO)
