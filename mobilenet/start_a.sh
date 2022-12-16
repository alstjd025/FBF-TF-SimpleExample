#!/bin/bash

mobilenet_bench="/home/xavier/FBF-TF-SimpleExample/mobilenet"
TflitePath="../../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/xavier/FBF-TF"


echo "TfLite mobilenet_bench Test"

. ${TflitePath}/build_aarch64_lib.sh
touch mobilenet_bench.cc
make

