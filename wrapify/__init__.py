__version__ = 0.4

from wrapify.utils import PluginRegistrar

PluginRegistrar.scan()

import logging
logging.getLogger().setLevel(logging.INFO)
