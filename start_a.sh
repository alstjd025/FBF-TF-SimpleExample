#!/bin/bash

UnitSimple="/home/xavier/Jetson/distributed"
TflitePath="../../tensorflow/tensorflow/lite/tools/make"
Tensorflowpath="home/xavier/tensorflow"


echo "TfLite Unit_simple Test"

. ${TflitePath}/build_aarch64_lib.sh
touch unit_simple.cc
make
.unit_simple mnist_original_102030.tflite

