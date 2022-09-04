# sampleMNIST

**Hello World** for TensorRT

## Description

This sample, sampleMNIST, is a simple hello workd example that performs the basic setup and initialization of TensorRT using the Caffe parser.

## How does this sample work?

This sample uses a Caffe model that was trained on the [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)

Specifically, this sample:

- Performs the basic setup and initialization of TensorRT using the Caffe parser
- Imports a trained Caffe model using Caffe parser
- Preprocesses the input and stores the result in a managed buffer
- Builds an engine
- Serializes and deserializes the engine
- Uses the engine to perform inference on an input image

To verify whether the engine is operating correctly, this sample picks a 28x28 image of a digit at random and runs inference on it using the engine it created. The output of the network is a probability distribution on the digit, showing which digit is likely that in the image

## TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documention

`Activation layer`: The Activation layer implements element-wise activation function.Specifically, this sample uses the Activation layer with the type `KRELU`

`Convolution layer`: The Convolution layer computes a 2D (channel, height, widht) convolution, with or without bias

`FullyConnected layer`: The FullyConnected layer implements a matrix-vector product, with or without bias.

`Pooling layer`: The Pooling layer implements pooling within a channel. Supported pooling types are `maximum, average, maximum-average blend`]

`Scale layer`: The scale layer implements a per-tensor, per-channel, orper-element affine transformation and/or exponentiation by constant values.

`SoftMax layer`: The SoftMax Layer applies the SoftMax function on the input tensor along an input dimension specified by the user.

## Preparing sample data

Download the sample data from `TensorRT release tarball`, if not already mounted under `/usr/src/tensorrt/data` (NVIDIA NGC containers) and set it to `$TRT_DATADIR`

```sh
export TRT_DATADIR = /usr/src/tensorrt/data
pushd $TRT_DATADIR/mnist
pip3 install Pillow
python3 download_pgm.py
popd
```

## Running the sample

1. Compile the sample by following build instruction in [TensorRT READEME](https://github.com/NVIDIA/TensorRT/)

2. Run the sample to perform inference on the digit

   ```sh
   ./sample_mnist [-h] [--datadir=/path/to/data/dir/] [--useDLA=N] [--fp16 or --int8]
   ```

   For example:

   ```sh
   ./sample_mnist --datadir $TRT_DATADIR/mnist --fp16
   ```

   This sample reads three Caffe files to build the network

   - `mnist.prototxt` The prototxt file that contains the network design.
   - `mnist.caffemodel` The model file which contains the trained weights for the network
   - `mnist_mean.binaryproto` The binaryproto file which contians the means.

   This sample can be run in FP16 and INT8 modes as well.

   **Note:** By default, the sample expects these files to be in either the `data/samples/mnist` or `data/mnist` directories. The list of default directories can be changed by adding one or more paths with `--datadir=/new/path` as a command line argument

3. Verify that the sample ran successfully. If the sample runs successfully you should see output simialr to the following; ASCll readering of the input iamge with digit3:

   ```sh
   &&&& RUNNING TensorRT.sample_mnist # ./sample_mnist
   [I] Building and running a GPU inference engine for MNIST
   [I] Input:
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@
   @@@@@@@@#-:.-=@@@@@@@@@@@@@@
   @@@@@%=     . *@@@@@@@@@@@@@
   @@@@% .:+%%%  *@@@@@@@@@@@@@
   @@@@+=#@@@@@# @@@@@@@@@@@@@@
   @@@@@@@@@@@%  @@@@@@@@@@@@@@
   @@@@@@@@@@@: *@@@@@@@@@@@@@@
   @@@@@@@@@@- .@@@@@@@@@@@@@@@
   @@@@@@@@@:  #@@@@@@@@@@@@@@@
   @@@@@@@@:   +*%#@@@@@@@@@@@@
   @@@@@@@%         :+*@@@@@@@@
   @@@@@@@@#*+--.::     +@@@@@@
   @@@@@@@@@@@@@@@@#=:.  +@@@@@
   @@@@@@@@@@@@@@@@@@@@  .@@@@@
   @@@@@@@@@@@@@@@@@@@@#. #@@@@
   @@@@@@@@@@@@@@@@@@@@#  @@@@@
   @@@@@@@@@%@@@@@@@@@@- +@@@@@
   @@@@@@@@#-@@@@@@@@*. =@@@@@@
   @@@@@@@@ .+%%%%+=.  =@@@@@@@
   @@@@@@@@           =@@@@@@@@
   @@@@@@@@*=:   :--*@@@@@@@@@@
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@
   
   [I] Output:
   0:
   1:
   2:
   3: **********
   4:
   5:
   6:
   7:
   8:
   9:
   
   &&&& PASSED TensorRT.sample_mnist # ./sample_mnist
   ```

   This output shows that the sample ran successfully; `PASSED`

