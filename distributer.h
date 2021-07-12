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
        virtual TfLiteStatus Invoke();
        virtual Interpreter* GetInterpreter();
    ~Distributer();
};

//Distributer Class for CPU
class DistributerCPU : public Distributer
{
    public:
        DistributerCPU();
        DistributerCPU(char* name, Interpreter* interpreter);
        TfLiteStatus Invoke();
        Interpreter* GetInterpreter();
           // {return interpreterCPU;}

        Interpreter* interpreterCPU;
        char* name;
};

//Distributer Class for GPU
class DistributerGPU : public Distributer
{
    public:
        DistributerGPU();
        DistributerGPU(char* name, Interpreter* interpreter);
        TfLiteStatus Invoke();
        Interpreter* GetInterpreter();
            //{return interpreterGPU;}

        Interpreter* interpreterGPU;
        char* name;
};

} // End of namespace tflite