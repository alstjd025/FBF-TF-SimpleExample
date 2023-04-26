#!/bin/bash

UnitSimple="/home/nvidia/FBF-TF-SimpleExample/other_bench"
TflitePath="../../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/nvidia/FBF-TF"


echo "TfLite Unit_simple Test"

. ${TflitePath}/build_aarch64_lib.sh
touch unit_simple.cc
make

