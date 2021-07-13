#pragma once
#include <ctime>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdarg>
#include <vector>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/c/common.h"
#include "thread"
#include "future"
#include "distributer.h"


#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

/*
Distributer handler class



*/


namespace tflite
{
class DistributerHandler
{
    private:
        std::vector<Distributer*> devices;
        tflite::InterpreterBuilder* builder_;
        int iDeviceCount; 
        const char* inputData;
        const char* fileName;
    public:
        DistributerHandler();
        DistributerHandler(const char* filename, const char* input_data);

        TfLiteStatus CreateDistributerCPU(const char* name);
        TfLiteStatus CreateDistributerGPU(const char* name);
        TfLiteStatus Invoke();

        void PrintInterpreterStatus();
        void PrintMsg(const char* msg);

        ~DistributerHandler() {};
        //tflite::Interpreter* GetInterpreter();
};

} //End of Namespace tflite