#!/bin/bash

UnitSimple="/home/odroid/FBF-TF-SimpleExample"
TflitePath="../tensorflow/tensorflow/lite/tools/make"
Tensorflowpath="home/odroid/tensorflow"


echo "TfLite Unit_simple Test"

. ${TflitePath}/build_bbb_lib.sh

echo "make Test Project"

touch ${UnitSimple}/unit_simple.cc

${UnitSimple}/ make

