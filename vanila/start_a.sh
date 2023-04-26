#!/bin/bash

UnitSimple="/home/nvidia/FBF-TF-SimpleExample/vanlia"
TflitePath="../../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/nvidia/FBF-TF"


echo "TfLite Vanila Test"

. ${TflitePath}/build_aarch64_lib.sh
touch vanila_tf.cc
make

