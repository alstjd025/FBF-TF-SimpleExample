#pragma once
#include <ctime>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdarg>
#include <vector>
#include <queue>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/c/common.h"
#include <functional>
#include "thread"
#include "mutex"
#include "future"
#include "unit.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

/*
Unit handler class
*/
namespace tflite
{
class UnitHandler
{
    private:
        std::vector<Unit*> vUnitContainer;
        tflite::InterpreterBuilder* builder_;
        int iUnitCount; 
        int numThreads;
        const char* inputData;
        const char* fileName;
        std::vector<std::function<void>*> workers;
    public:
        UnitHandler();
        UnitHandler(const char* filename, const char* input_data);

        TfLiteStatus CreateUnitCPU(const char* name);
        TfLiteStatus CreateUnitGPU(const char* name);
        TfLiteStatus Invoke(std::vector<cv::Mat> vec);

        void PrintInterpreterStatus();
        void PrintMsg(const char* msg);

        ~UnitHandler() {};
        //tflite::Interpreter* GetInterpreter();
};

} //End of Namespace tflite