import sys
import os

def setup(app):
    app.add_css_file('wide_theme.css')

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
master_doc = 'index'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static'] 
html_css_files = ['wide_theme.css'] 

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'myst_parser']
# run from within an environment that has all requirements installed besides ROS2
autodoc_mock_imports = ["rclpy", "rclpy.node", "Parameter", "Node"]
source_suffix = ['.rst', '.md']
exclude_patterns = ["_build"]
# autodoc_mock_imports = ["rclpy", "rospy", "yarp", "cv2", "numpy", "yaml",
#                         "torch", "pandas", "tensorflow", "jax", "jaxlib", "mxnet", "paddle"]
sys.path.insert(0, os.path.abspath('../'))
