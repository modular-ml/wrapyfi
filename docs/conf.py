import sys
import os

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
master_doc = 'usage'
html_theme = 'sphinx_rtd_theme'
extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'myst_parser']
autodoc_mock_imports = ["rclpy", "rclpy.node", "Parameter", "Node"]
source_suffix = ['.rst', '.md']
# autodoc_mock_imports = ["rclpy", "rospy", "yarp", "cv2", "numpy", "yaml",
#                         "torch", "pandas", "tensorflow", "jax", "jaxlib", "mxnet", "paddle"]
sys.path.insert(0, os.path.abspath('../'))