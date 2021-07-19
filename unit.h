#pragma once
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdarg>
#include <string>
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/c/common.h"
#include "mutex"
#include "thread"
#include "future"

#define Image_x 28
#define Image_y 28
#define Image_ch 1
#define SEQ 10000
#define OUT_SEQ 100

extern std::mutex mtx_lock;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

/*
Unit Class for Tflite Distribute Stradegy

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


class Unit 
{   
    public:
        virtual Interpreter* GetInterpreter() = 0;
        virtual TfLiteStatus Invoke() = 0;
        virtual void SetInput(std::vector<cv::Mat> input_) = 0;

        std::vector<cv::Mat> input;
        std::thread myThread;
        std::unique_ptr<tflite::Interpreter> interpreter;
        std::string name;
};

//Unit Class for CPU
class UnitCPU : public Unit
{
    public:
        UnitCPU();
        UnitCPU(const char* name_, std::unique_ptr<tflite::Interpreter>* interpreter);
        ~UnitCPU() {};
        TfLiteStatus Invoke();
        Interpreter* GetInterpreter();
        void SetInput(std::vector<cv::Mat> input_);
        
        std::vector<cv::Mat> input;
        std::thread myThread;
        std::unique_ptr<tflite::Interpreter>* interpreterCPU;
        std::string name;
        double cpu_t;
};

//Unit Class for GPU
class UnitGPU : public Unit
{
    public:
        UnitGPU();
        UnitGPU(const char* name, std::unique_ptr<tflite::Interpreter>* interpreter);
        ~UnitGPU() {};
        TfLiteStatus Invoke();
        Interpreter* GetInterpreter();
        void SetInput(std::vector<cv::Mat> input_);

        std::vector<cv::Mat> input;
        std::thread myThread;
        std::unique_ptr<tflite::Interpreter>* interpreterGPU;
        std::string name;
        double gpu_t;
};

} // End of namespace tflite