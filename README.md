TensorRT Python Sample for a Re-Trained SSD MobileNet V2 Model (only faces' detection)
======================================
### Original NVIDIA sample's GiHub repository: <a href=https://github.com/AastaNV/TRT_object_detection>AastaNV/TRT_object_detection</a>

### Original Jeroen Bédorf's tutorial: <a href=https://www.minds.ai/post/deploying-ssd-mobilenet-v2-on-the-nvidia-jetson-and-nano-platforms>Deploying SSD mobileNet V2 on the NVIDIA Jetson and Nano platforms</a>
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

## Install Tensorflow 1 dependencies and PyCUDA

```C
$ sudo apt-get update
$ sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
$ sudo apt-get install python3-pip
$ pip3 install -U pip testresources setuptools==49.6.0
$ pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
$ pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 'tensorflow<2'
$ pip3 install numpy pycuda --user
```

</br>
</br>

## Prepare your pre-trained model

The base object detection model is available here: <a href=https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>TensorFlow model zoo</a>.
</br>
**Remember that this sample is adjusted only for re-trained SSD MobileNet V2 models (use the frozen_inference_graph.pb file, exported after your custom training).**
</br>
If original sample is required, visit: <a href=https://github.com/AastaNV/TRT_object_detection>AastaNV/TRT_object_detection</a>

```C
$ git clone https://github.com/brokenerk/TRT-SSD-MobileNetV2.git
$ cd TRT-SSD-MobileNetV2
$ mkdir model
$ cp [model].tar.gz model/
$ tar zxvf model/[model].tar.gz -C model/
// ============================================================================
// Or just put your frozen_inference_graph.pb file inside the model/ directory
// ============================================================================
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
