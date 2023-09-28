import os
import re


def get_project_info_from_setup():
    curr_dir = os.path.dirname(__file__)
    setup_path = os.path.join(curr_dir, '..', 'setup.py')
    with open(setup_path, 'r') as f:
        content = f.read()
    
    name_match = re.search(r"name\s*=\s*['\"]([^'\"]*)['\"]", content)
    version_match = re.search(r"version\s*=\s*['\"]([^'\"]*)['\"]", content)
    url_match = re.search(r"url\s*=\s*['\"]([^'\"]*)['\"]", content)
    
    if not name_match or not version_match or not url_match:
        raise RuntimeError("Unable to find name, version, or url string.")
        
    return {
        'name': name_match.group(1),
        'version': version_match.group(1),
        'url': url_match.group(1)
    }

# Extract project info
project_info = get_project_info_from_setup()

__version__ = project_info['version']
__url__ = project_info['url']
name = project_info['name']

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
