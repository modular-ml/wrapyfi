import os
import re


def get_project_info_from_setup():
    try:
        curr_dir = os.path.dirname(__file__)
        setup_path = os.path.join(curr_dir, '..', 'setup.py')
        with open(setup_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return {}
    name_match = re.search(r"name\s*=\s*['\"]([^'\"]*)['\"]", content)
    version_match = re.search(r"version\s*=\s*['\"]([^'\"]*)['\"]", content)
    url_match = re.search(r"url\s*=\s*['\"]([^'\"]*)['\"]", content)
    
    if not name_match or not version_match or not url_match:
        # raise RuntimeError("Unable to find name, version, or url string.")
        return {}
        
    return {
        'name': name_match.group(1),
        'version': version_match.group(1),
        'url': url_match.group(1)
    }


# extract project info
project_info = get_project_info_from_setup()

__version__ = project_info.get('version', None)
__url__ = project_info.get('url', None)
name = project_info.get('name', 'wrapyfi')

if __version__ is None or __url__ is None:
    try:
        from importlib import metadata
        __version__ = metadata.version(__name__)
        __url__ = metadata.metadata(__name__)["Home-page"]
        if __url__ is None:
            for url_extract in metadata.metadata("wrapyfi").get_all('Project-URL'):
                __url__ = url_extract.split(", ")[1] if url_extract.split(", ")[0] == "Homepage" else __url__
    except ImportError:
        try:
            import pkg_resources
            __version__ = pkg_resources.require(name)[0].version
            __url__ = pkg_resources.get_distribution(__name__).metadata["Home-page"]
        except pkg_resources.DistributionNotFound:
            __version__ = "unknown_version"
            __url__ = "unknown_url"
    except Exception:
        __version__ = "unknown_version"
        __url__ = "unknown_url"

from wrapyfi.utils import PluginRegistrar

PluginRegistrar.scan()

import logging
logging.getLogger().setLevel(logging.INFO)
