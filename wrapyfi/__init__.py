import os
import re


def get_project_info_from_setup():
    # when Wrapyfi is not installed
    try:
        curr_dir = os.path.dirname(__file__)
        setup_path = os.path.join(curr_dir, "..", "setup.py")
        with open(setup_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return {}
    name_match = re.search(r"name\s*=\s*['\"]([^'\"]*)['\"]", content)
    version_match = re.search(r"version\s*=\s*['\"]([^'\"]*)['\"]", content)
    url_match = re.search(r"url\s*=\s*['\"]([^'\"]*)['\"]", content)
    doc_match = re.search(r"'Documentation':\s*['\"]([^'\"]*)['\"]", content)
    
    if not name_match or not version_match or not url_match:
        # raise RuntimeError("Unable to find name, version, or url string.")
        return {}
        
    return {
        'name': name_match.group(1),
        'version': version_match.group(1),
        'url': url_match.group(1),
        'doc': None if not doc_match else doc_match.group(1)
    }


# extract project info
project_info = get_project_info_from_setup()

__version__ = project_info.get('version', None)
__url__ = project_info.get('url', None)
__doc__ = project.info.get('doc', None)
name = project_info.get('name', 'wrapyfi')

if __version__ is None or __url__ is None or __doc__ is None:
    try:
        from importlib import metadata
        mdata = metadata.metadata(__name__)
        __version__ = metadata.version(__name__)
        __url__ = mdata["Home-page"]
        # when installed with PyPi
        if __url__ is None:
            for url_extract in mdata.get_all("Project-URL"):
                __url__ = url_extract.split(", ")[1] if url_extract.split(", ")[0] == "Homepage" else __url__
        if __doc__ is None:
            for url_extract in mdata.get_all("Project-URL"):
                __doc__ = url_extract.split(", ")[1] if url_extract.split(", ")[0] == "Documentation" else __url__
    except ImportError:
        try:
            # when Python < 3.8 and setuptools/pip have not been updated
            import pkg_resources
            mdata = pkg_resources.get_distribution(__name__).metadata
            __version__ = pkg_resources.require(__name__)[0].version
            __url__ = mdata["Home-page"]
            # when installed with PyPi
            if __url__ is None:
                for url_extract in mdata.get_all("Project-URL"):
                    __url__ = url_extract.split(", ")[1] if url_extract.split(", ")[0] == "Homepage" else __url__
            if __doc__ is None:
                for url_extract in mdata.get_all("Project-URL"):
                    __doc__ = url_extract.split(", ")[1] if url_extract.split(", ")[0] == "Documentation" else __url__
        except pkg_resources.DistributionNotFound:
            __version__ = "unknown_version"
            __url__ = "unknown_url"
            __doc__ = "unknown_url"
    except Exception:
        __version__ = "unknown_version"
        __url__ = "unknown_url"
        __doc__ = "unknown_url"

from wrapyfi.utils import PluginRegistrar

PluginRegistrar.scan()

import logging
logging.getLogger().setLevel(logging.INFO)
