import base64
import json

from wrapyfi.utils import *

try:
    from pint import Quantity
    HAVE_PINT = True
except ImportError:
    HAVE_PINT = False


@PluginRegistrar.register(types=None if not HAVE_PINT else Quantity.__mro__[:-1])
class PintData(Plugin):
    def __init__(self, **kwargs):
        pass  # Reverted the UnitRegistry initialization here

    def encode(self, obj, *args, **kwargs):
        if isinstance(obj, Quantity):
            obj_type = 'Quantity'
            obj_data = json.dumps({
                'magnitude': obj.magnitude,
                'units': str(obj.units)
            }).encode('ascii')
            obj_data = base64.b64encode(obj_data).decode('ascii')
        return True, dict(__wrapyfi__=(str(self.__class__.__name__), obj_data, obj_type))

    def decode(self, obj_type, obj_full, *args, **kwargs):
        obj_data = base64.b64decode(obj_full[1].encode('ascii')).decode('ascii')
        obj_data = json.loads(obj_data)
        obj_type = obj_full[2]
        if obj_type == 'Quantity':
            from pint import UnitRegistry
            ureg = UnitRegistry()
            obj = Quantity(obj_data['magnitude'], ureg.parse_expression(obj_data['units']))
        return True, obj