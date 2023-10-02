## Plugins

The **NativeObject** message type supports structures beyond native python objects. Wrapyfi already supports a number of non-native objects including numpy arrays and tensors. Wrapyfi can be extended to support objects by using the plugin API. All currently supported plugins by Wrapyfi can be found in the [plugins directory](../wrapyfi/plugins). Plugins can be added by:
* Creating a derived class that inherits from the base class `wrapyfi.utils.Plugin`
* Overriding the `encode` method for converting the object to a `json` serializable string. Deserializing the string is performed within the overridden `decode` method
* Specifying custom object properties by defining keyword arguments for the class constructor. These properties can be passed directly to the Wrapyfi decorator
* Decorating the class with `@PluginRegistrar.register` and appending the plugin to the list of supported objects
* Appending the script path where the class is defined to the `WRAPYFI_PLUGINS_PATH` environment variable
* Ensure that the plugin resides within a directory named `plugins` residing inside the `WRAPYFI_PLUGINS_PATH` and that the directory contains an `__init__.py` file

```{warning}
Due to differences in versions, the decoding may result in inconsitent outcomes, which must be handled for all versions e.g., MXNet plugin differences are handled in the existing plugin. 
```

### Data Structure Types

Other than native python objects, the following objects are supported:

* `numpy.ndarray` and `numpy.generic`
* `pandas.DataFrame` and `pandas.Series` (pandas v1)
* `torch.Tensor`
* `tensorflow.Tensor` and `tensorflow.EagerTensor`
* `mxnet.nd.NDArray`
* `jax.numpy.DeviceArray`
* `paddle.Tensor`
* `PIL.Image`
* `pyarrow.StructArray`
* `xarray.DataArray` and `xarray.Dataset`

### Device Mapping for Tensors

To map tensor listener decoders to specific devices (CPUs/GPUs), add an argument to tensor data structures with direct GPU/TPU mapping to support re-mapping on mirrored nodes e.g.,

```
@PluginRegistrar.register
class MXNetTensor(Plugin):
    def __init__(self, load_mxnet_device=None, map_mxnet_devices=None, **kwargs):
```

where `map_mxnet_devices` should be `{'default': mxnet.gpu(0)}` when `load_mxnet_device=mxnet.gpu(0)` and `map_mxnet_devices=None`.
For instance, when `load_mxnet_device=mxnet.gpu(0)` or `load_mxnet_device="cuda:0"`, `map_mxnet_devices` can be set manually as a dictionary representing the source device as key and the target device as value for non-default device maps. 

Suppose we have the following Wrapyfied method:

```

    @MiddlewareCommunicator.register("NativeObject", args.mware, "Notify", "/notify/test_native_exchange",
                                     carrier="tcp", should_wait=True, load_mxnet_device=mxnet.cpu(0), 
                                     map_mxnet_devices={"cuda:0": "cuda:1", 
                                                         mxnet.gpu(1): "cuda:0", 
                                                         "cuda:3": "cpu:0", 
                                                         mxnet.gpu(2):  mxnet.gpu(0)})
    def exchange_object(self):
        msg = input("Type your message: ")
        ret = {"message": msg,
               "mx_ones": mxnet.nd.ones((2, 4)),
               "mxnet_zeros_cuda1": mxnet.nd.zeros((2, 3), ctx=mxnet.gpu(1)),
               "mxnet_zeros_cuda0": mxnet.nd.zeros((2, 3), ctx=mxnet.gpu(0)),
               "mxnet_zeros_cuda2": mxnet.nd.zeros((2, 3), ctx=mxnet.gpu(2)),
               "mxnet_zeros_cuda3": mxnet.nd.zeros((2, 3), ctx=mxnet.gpu(3))}
        return ret,
```

then the source and target gpus 1 & 0 would be flipped, gpu 3 would be placed on cpu 0, and gpu 2 would be placed on gpu 0. Defining `mxnet.gpu(1): mxnet.gpu(0)` and `cuda:1`: `cuda:2` in the same mapping should raise an error since the same device is mapped to two different targets.

The plugins supporting remapping are:

* `mxnet.nd.NDArray`
* `torch.Tensor`
* `paddle.Tensor`

### Serialization

```{warning}
When encoding dictionaries, `json` supports string keys only and converts any instances of `int` keys to string, causing a difference between the publisher and subscriber returns. It is best to avoid using `int` keys, otherwise handle the difference on the receiving end.
```

Wrapyfi currently supports JSON as the only serializer. This introduces a number of limitations (beyond serializing native python objects only by default), including:

* dictionary keys cannot be integers. Integers are automatically converted to strings
* Tuples are converted to lists. Sets are not serializable. Tuples and sets are encoded as strings and restored on listening, which resolves this limitation but adds to the encoding overhead. This conversion is supported in Wrapyfi

