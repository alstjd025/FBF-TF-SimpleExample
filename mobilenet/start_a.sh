#!/bin/bash

mobilenet_bench="/home/nvidia/FBF-TF-SimpleExample/mobilenet"
TflitePath="../../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/nvidia/FBF-TF"


echo "TfLite mobilenet_bench Test"

. ${TflitePath}/build_aarch64_lib.sh
touch mobilenet_bench.cc
make

