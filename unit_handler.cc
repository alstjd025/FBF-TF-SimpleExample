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
    model = new std::unique_ptr<tflite::FlatBufferModel>\
    (tflite::FlatBufferModel::BuildFromFile(filename));
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

TfLiteStatus UnitHandler::CreateUnitCPU(tflite::UnitType eType,
                                         std::vector<cv::Mat> input){
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

TfLiteStatus UnitHandler::CreateAndInvokeCPU(tflite::UnitType eType,
                                             std::vector<cv::Mat> input){ 
    mtx_lock.lock();
    if (CreateUnitCPU(eType, input) != kTfLiteOk){
        PrintMsg("CreateUnitCPUError");
        return kTfLiteError;
    }
    mtx_lock.unlock();
    std::vector<Unit*>::iterator iter;
    for(iter = vUnitContainer.begin(); iter != vUnitContainer.end(); ++iter){
        if((*iter)->GetUnitType() == eType){
            if((*iter)->Invoke() != kTfLiteOk)
                return kTfLiteError;
        }
    }
    return kTfLiteOk;
}

TfLiteStatus UnitHandler::CreateUnitGPU(tflite::UnitType eType,
                                         std::vector<cv::Mat> input){
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
        .experimental_flags = 4,
        .max_delegated_partitions = 30,
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
    //Set ContextHandler Pointer
    vUnitContainer.push_back(temp);
    iUnitCount++;
    PrintMsg("Build GPU Interpreter");
    PrintMsg("GPU Interpreter Pre Invoke State");
    tflite::PrintInterpreterState(interpreter->get());
    return kTfLiteOk;
}


TfLiteStatus UnitHandler::CreateAndInvokeGPU(tflite::UnitType eType,
                                             std::vector<cv::Mat> input){
    mtx_lock.lock();
    if (CreateUnitGPU(eType, input) != kTfLiteOk){
        PrintMsg("CreateUnitCPUError");
        return kTfLiteError;
    }
    mtx_lock.unlock();
    std::vector<Unit*>::iterator iter;
    for(iter = vUnitContainer.begin(); iter != vUnitContainer.end(); ++iter){
        if((*iter)->GetUnitType() == eType){
           if((*iter)->Invoke() != kTfLiteOk)
               return kTfLiteError;
        }
    }
    return kTfLiteOk;
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

TfLiteStatus UnitHandler::Invoke(tflite::UnitType eType, tflite::UnitType eType_,
                                 std::vector<cv::Mat> input){
    PrintMsg("Invoke");
    //std::thread cpu;
    std::thread gpu;
    //cpu = std::thread(&UnitHandler::CreateAndInvokeCPU, this, eType, input);
    gpu = std::thread(&UnitHandler::CreateAndInvokeGPU, this, eType_, input);

    //cpu.join();
    gpu.join();

}

TfLiteStatus UnitHandler::ContextHandler(tflite::UnitType eType, TfLiteContext* context){
    if(eType != UnitType::CPU0){
        sharedContext* slaveData = CreateSharedContext(eType, context);
        if(PushTensorContextToQueue(slaveData) != kTfLiteOk){
            PrintMsg("Context Pushing Error");
            return kTfLiteError;
        }
        return kTfLiteOk;
    }
    else{
        if(ConcatContext(context, PopTensorContextFromQueue()) != kTfLiteOk){
            PrintMsg("Context Concat Error");
            return kTfLiteError;
        }
        return kTfLiteOk;
    }
}

sharedContext* UnitHandler::CreateSharedContext(tflite::UnitType eType, TfLiteContext* context){
    return new sharedContext{context, eType};
}

TfLiteStatus UnitHandler::ConcatContext(TfLiteContext* context, sharedContext* PopedData){
    //Concate original context and PopedData
    //Concat info initiallizing function needs to be implemented Later
    return kTfLiteOk;
}

TfLiteStatus UnitHandler::PushTensorContextToQueue(sharedContext* slaveData){
    if(slaveData != nullptr){
        mtx_lock.lock();
        qSharedData->push(slaveData);
        mtx_lock.unlock();
        return kTfLiteOk;
    }
    PrintMsg("SlaveData Error");
    return kTfLiteError;
}

sharedContext* UnitHandler::PopTensorContextFromQueue(){
    sharedContext* temp;
    while(!qSharedData->empty()){
        mtx_lock.lock();
        temp = qSharedData->front();
        qSharedData->pop();
        mtx_lock.unlock();
    }
    return temp;
}




} // End of namespace tflite
