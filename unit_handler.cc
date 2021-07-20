#include "unit_handler.h"
#include <typeinfo>

extern std::mutex mtx_lock;

namespace tflite
{

UnitHandler::UnitHandler() :  fileName(nullptr), builder_(nullptr) {}

UnitHandler::UnitHandler(const char* filename)
                                        :fileName(filename)
{
    std::cout << "You have " << std::thread::hardware_concurrency() << " Processors " << "\n";
    vUnitContainer.reserve(10);
    std::unique_ptr<tflite::FlatBufferModel>* model;
    model = new std::unique_ptr<tflite::FlatBufferModel>(tflite::FlatBufferModel::BuildFromFile(filename));
    TFLITE_MINIMAL_CHECK(model != nullptr);
    // Build the interpreter with the InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver* resolver;
    resolver = new tflite::ops::builtin::BuiltinOpResolver;
    builder_ = new tflite::InterpreterBuilder(**model, *resolver);
    PrintMsg("Create InterpreterBuilder");
    if(builder_ == nullptr){
        PrintMsg("InterpreterBuilder nullptr ERROR");
    }
}

TfLiteStatus UnitHandler::CreateUnitCPU(tflite::UnitType eType, std::vector<cv::Mat> input){
    if(builder_ == nullptr){
        PrintMsg("InterpreterBuilder nullptr ERROR");
        return kTfLiteError;
    }
    std::unique_ptr<tflite::Interpreter>* interpreter;
    interpreter = new std::unique_ptr<tflite::Interpreter>;
    (*builder_)(interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    TFLITE_MINIMAL_CHECK(interpreter->get()->AllocateTensors() == kTfLiteOk);
    
    UnitCPU* temp;
    temp = new UnitCPU(eType, std::move(interpreter));
    temp->SetInput(input);
    vUnitContainer.push_back(temp);
    iUnitCount++;    
    PrintMsg("Build CPU Interpreter");
    return kTfLiteOk;
}

TfLiteStatus UnitHandler::CreateUnitCPUandInvoke(tflite::UnitType eType, std::vector<cv::Mat> input){
    if (CreateUnitCPU(eType, input) != kTfLiteOk){
        PrintMsg("CreateUnitCPUError");
        return kTfLiteError;
    }
    std::vector<Unit*>::iterator iter;
    for(iter = vUnitContainer.begin(); iter != vUnitContainer.end(); ++iter){
        if((*iter)->GetType() == UnitType::CPU0){
            if((*iter)->Invoke() != kTfLiteOk){
                return kTfLiteError;
        }
    }
    return kTfLiteOk;
}

TfLiteStatus UnitHandler::CreateUnitGPU(tflite::UnitType eType, std::vector<cv::Mat> input){
    if(builder_ == nullptr){
        PrintMsg("InterpreterBuilder nullptr ERROR");
        return kTfLiteError;
    }
    std::unique_ptr<tflite::Interpreter>* interpreter;
    interpreter = new std::unique_ptr<tflite::Interpreter>;
    (*builder_)(interpreter);
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
    if(interpreter->get()->ModifyGraphWithDelegate(MyDelegate) != kTfLiteOk) {
        PrintMsg("Unable to Use GPU Delegate");
        return kTfLiteError;
    }
    TFLITE_MINIMAL_CHECK(interpreter->get()->AllocateTensors() == kTfLiteOk);
    UnitGPU* temp;
    temp = new UnitGPU(eType, std::move(interpreter));
    temp->SetInput(input);
    vUnitContainer.push_back(temp);
    iUnitCount++;
    PrintMsg("Build GPU Interpreter");
    return kTfLiteOk;
}

TfLiteStatus UnitHandler::CreateUnitGPUandInvoke(tflite::UnitType eType, std::vector<cv::Mat> input){
    if (CreateUnitCPU(eType, input) != kTfLiteOk){
        PrintMsg("CreateUnitCPUError");
        return kTfLiteError;
    }
    std::vector<Unit*>::iterator iter;
    for(iter = vUnitContainer.begin(); iter != vUnitContainer.end(); ++iter){
        if((*iter)->GetType() == UnitType::CPU0){
            if((*iter)->Invoke() != kTfLiteOk){
                return kTfLiteError;
        }
    }
}

void UnitHandler::PrintMsg(const char* msg){
    std::cout << "UnitHandler : \"" << msg << "\"\n";
    return;
}

void UnitHandler::PrintInterpreterStatus(){
    std::vector<Unit*>::iterator iter;
    for(iter = vUnitContainer.begin(); iter != vUnitContainer.end(); ++iter){
        //std::cout << "Device ====== " << (*iter)->eType << " ======\n";
        PrintInterpreterState((*iter)->GetInterpreter());
    }
    return;
}

TfLiteStatus UnitHandler::Invoke(){
    PrintMsg("Invoke");
    std::vector<Unit*>::iterator iter;
    for(iter = vUnitContainer.begin(); iter != vUnitContainer.end(); ++iter){
        (*iter)->myThread = std::thread(&Unit::Invoke, (*iter));
    }
}


} // End of namespace tflite