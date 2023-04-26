#!/bin/bash

UnitSimple="/home/nvidia/FBF-TF-SimpleExample/new_bench"
TflitePath="../../../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/nvidia/FBF-TF"


echo "TfLite scheduler Test"

. ${TflitePath}/build_aarch64_lib.sh
touch scheduler.cc
make

