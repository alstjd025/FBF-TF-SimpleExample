#include "distributer_handler.h"
#include <typeinfo>

namespace tflite
{

DistributerHandler::DistributerHandler() : inputData(nullptr), 
                                        fileName(nullptr), builder_(nullptr) {}

DistributerHandler::DistributerHandler(const char* filename, const char* input_data)
                                        :inputData(input_data), fileName(filename)
{
    devices.reserve(10);
    std::unique_ptr<tflite::FlatBufferModel> model = 
    tflite::FlatBufferModel::BuildFromFile(filename);
    TFLITE_MINIMAL_CHECK(model != nullptr);
    // Build the interpreter with the InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    builder_ = new tflite::InterpreterBuilder(*model, resolver);
    PrintMsg("Create InterpreterBuilder");
    if(builder_ == nullptr){
        PrintMsg("InterpreterBuilder nullptr ERROR");
    }
}

TfLiteStatus DistributerHandler::Invoke(){

}


TfLiteStatus DistributerHandler::CreateDistributerCPU(char* name){
    if(builder_ == nullptr){
        PrintMsg("InterpreterBuilder nullptr ERROR");
        return kTfLiteError;
    }
    std::unique_ptr<tflite::Interpreter> interpreter;
    (*builder_)(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    DistributerCPU* temp;
    temp = new DistributerCPU(name, std::move(interpreter));
    devices.push_back(temp);
    iDeviceCount++;
    PrintMsg("Build CPU Interpreter");
    return kTfLiteOk;
}

TfLiteStatus DistributerHandler::CreateDistributerGPU(char* name){
    if(builder_ == nullptr){
        PrintMsg("InterpreterBuilder nullptr ERROR");
        return kTfLiteError;
    }
    std::unique_ptr<tflite::Interpreter> interpreter;
    (*builder_)(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    TfLiteDelegate *MyDelegate = NULL;
    const TfLiteGpuDelegateOptionsV2 options = {
        .is_precision_loss_allowed = 1, 
        .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
        .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
        .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
        .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
    };
    MyDelegate = TfLiteGpuDelegateV2Create(&options);

    if(interpreter->ModifyGraphWithDelegate(MyDelegate) != kTfLiteOk) {
        PrintMsg("Unable to Use GPU Delegate");
        return kTfLiteError;
    }
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    DistributerGPU* temp;
    temp = new DistributerGPU(name, std::move(interpreter));
    devices.push_back(temp);
    iDeviceCount++;
    PrintMsg("Build GPU Interpreter");
    return kTfLiteOk;
}

void DistributerHandler::PrintMsg(const char* msg){
    std::cout << "DistributerHandler : \"" << msg << "\"\n";
    return;
}

void DistributerHandler::PrintInterpreterStatus(){
    std::vector<Distributer*>::iterator iter;
    for(iter = devices.begin(); iter != devices.end(); ++iter){
            std::cout << "Device =="<< (*iter)->name << "==\n";
            std::cout << "Name   =="<< (*iter)->type << "==\n";
           PrintInterpreterState((*iter)->GetInterpreter());
        }
    return;
}

} // End of namespace tflite