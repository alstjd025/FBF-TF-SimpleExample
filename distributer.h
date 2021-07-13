#pragma once
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdarg>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/c/common.h"
#include "thread"
#include "future"


/*
Distributer Class for Tflite Distribute Stradegy

Class Constructor
    args -> (tflite::interpreterBuilder *builder,
            input Data(cv::Mat),
            Num of Devices,
            Device Name(ex. GPU1), 
            Device Type(GPU or CPU),
            .
            .
            )
    Make Interpreter & TfLiteDelegate Object for each Device
    Delegate Interpreter & Allocate Tensors
    
    **IMPORTANT**
    YOU BASICALLY HAVE ONE CPU TfLiteInterpreter Object

*/

namespace tflite{

class Distributer 
{   
    public:
        //virtual TfLiteStatus Invoke() = 0;
        virtual Interpreter* GetInterpreter() = 0;
        //virtual ~Distributer() = 0;
};

//Distributer Class for CPU
class DistributerCPU : public Distributer
{
    public:
        DistributerCPU();
        DistributerCPU(const char* name, Interpreter* interpreter);
        ~DistributerCPU() {};
        //TfLiteStatus Invoke();
        Interpreter* GetInterpreter();

        Interpreter* interpreterCPU;
        const char* name;
};

//Distributer Class for GPU
class DistributerGPU : public Distributer
{
    public:
        DistributerGPU();
        DistributerGPU(const char* name, Interpreter* interpreter);
        ~DistributerGPU() {};
        //TfLiteStatus Invoke();
        Interpreter* GetInterpreter();

        Interpreter* interpreterGPU;
        const char* name;
};

} // End of namespace tflite