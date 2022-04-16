#!/bin/bash

UnitSimple="/home/xavier/FBF-TF-SimpleExample"
TflitePath="../FBF-TF/tensorflow/lite/tools/make"
Tensorflowpath="home/xavier/FBF-TF"


echo "TfLite Unit_simple Test"

. ${TflitePath}/build_aarch64_lib.sh
touch unit_simple.cc
make

