from docutils import nodes
from sphinx.transforms import SphinxTransform
import re

REPLACEMENTS = {
    'https://github.com/fabawi/wrapyfi/tree/main/wrapyfi_extensions/yarp/README.md':
        'yarp_install_lnk.html',
    'https://github.com/modular-ml/wrapyfi_ros2_interfaces/blob/master/README.md':
        'ros2_interfaces_lnk.html',
    'https://github.com/modular-ml/wrapyfi_ros_interfaces/blob/master/README.md':
        'ros_interfaces_lnk.html',
    'https://github.com/fabawi/wrapyfi/tree/main/dockerfiles/README.md':
        'wrapyfi_docker_lnk.html',
}

class LinkModifier(SphinxTransform):
    default_priority = 999

    def apply(self):
        for node in self.document.traverse(nodes.reference):
            uri = node.get('refuri', '')
            for link in REPLACEMENTS.keys():
                if link in uri:
                    # Extract rank value
                    match = re.search(r'\?rank=(-?\d+)', uri)
                    if match:
                        rank = int(match.group(1))
                        if rank < 0:
                            prefix = '../' * -rank
                            new_uri = prefix + REPLACEMENTS[link]
                        else:
                            new_uri = REPLACEMENTS[link]
                    else:
                        # Default to rank -2 if no rank parameter
                        new_uri = '../../' + REPLACEMENTS[link]

                    node['refuri'] = uri.replace(link, new_uri)

def setup(app):
    app.add_transform(LinkModifier)
