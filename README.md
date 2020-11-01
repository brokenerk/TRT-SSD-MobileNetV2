TensorRT Python Sample for a Re-Trained SSD MobileNet V2 Model (only Faces detection)
======================================
### Original GiHub repository: <a href=https://github.com/AastaNV/TRT_object_detection>AastaNV/TRT_object_detection</a>
</br>

**Tested on a NVIDIA Jetson AGX Xavier with Jetpack 4.3 and Tensorflow 1.15.**

</br>

Performance includes memcpy and inference.
</br>

| Model | Input Size | TRT Nano |
|:------|:----------:|-----------:|
| ssd_mobilenet_v2_coco | 300x300 | 46ms |

Since the optimization of preprocessing is not ready yet, image read/write time is not included here.
</br>
</br>

## Install dependencies

```C
$ sudo apt-get install python3-pip libhdf5-serial-dev hdf5-tools
$ pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 'tensorflow<2'
$ pip3 install numpy pycuda --user
```

</br>
</br>

## Prepare your pre-trained model

The base object detection model is available here: <a href=https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>TensorFlow model zoo</a>.
</br>
**Remember that this sample is adjusted only for re-trained (transfer learning) SSD MobileNet V2 models.**
</br>
If original sample is required, visit: <a href=https://github.com/AastaNV/TRT_object_detection>AastaNV/TRT_object_detection</a>

```C
$ git clone https://github.com/brokenerk/TRT-SSD-Fixed.git
$ cd TRT-SSD-Fixed
$ mkdir model
$ cp [model].tar.gz model/
$ tar zxvf model/[model].tar.gz -C model/
```

##### Supported models:

- ssd_mobilenet_v2_coco

</br>
</br>

## Update graphsurgeon converter

Edit /usr/lib/python3.6/dist-packages/graphsurgeon/node_manipulation.py

```C
def create_node(name, op=None, _do_suffix=False, **kwargs):
     node = NodeDef()
     node.name = name
     node.op = op if op else name
     node.attr["dtype"].type = 1
     for key, val in kwargs.items():
         if key == "dtype":
             node.attr["dtype"].type = val.as_datatype_enum
```
</br>
</br>

## RUN

**1. Maximize the Nano performance**
```C
$ sudo nvpmodel -m 0
$ sudo jetson_clocks
```
</br>

**2. Execute**
```C
$ python3 main.py [test_image_path]
```

It takes some time to compile a TensorRT model when the first launching.
</br>
After that, TensorRT engine can be created directly with the serialized .bin file
</br>
</br>
@ To get more memory, it's recommended to turn-off X-server.
</br>
</br>
</br>
</br>
</br>
