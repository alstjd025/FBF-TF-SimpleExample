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
#include "future"
#include "unit.h"

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
        std::vector<cv::Mat> inputData;
        const char* fileName;
        std::vector<std::function<void>*> workers;
    public:
        UnitHandler();
        UnitHandler(const char* filename);

        TfLiteStatus CreateUnitCPU(tflite::UnitType eType, std::vector<cv::Mat> input);
        TfLiteStatus CreateUnitGPU(tflite::UnitType eType, std::vector<cv::Mat> input);
        TfLiteStatus Invoke(tflite::UnitType eType, tflite::UnitType eType_, std::vector<cv::Mat> input);

        TfLiteStatus CreateAndInvokeCPU(tflite::UnitType eType, std::vector<cv::Mat> input);
        TfLiteStatus CreateAndInvokeGPU(tflite::UnitType eType, std::vector<cv::Mat> input);

        void PrintInterpreterStatus();
        void PrintMsg(const char* msg);

        ~UnitHandler() {};
        //tflite::Interpreter* GetInterpreter();
};

} //End of Namespace tflite

