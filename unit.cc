#include "unit.h"

namespace tflite
{

// UnitCPU
UnitCPU::UnitCPU() : name("NONE"), interpreterCPU(nullptr){}

UnitCPU::UnitCPU(const char* name_, std::unique_ptr<tflite::Interpreter> interpreter) 
            : name(name_), interpreterCPU(std::move(interpreter)) {
    myThread = std::thread(&UnitCPU::Invoke, this);
}

TfLiteStatus UnitCPU::Invoke() {
    return kTfLiteOk;
}

Interpreter* UnitCPU::GetInterpreter(){return interpreterCPU.get();}




// UnitGPU
UnitGPU::UnitGPU() : name("NONE"), interpreterGPU(nullptr){}

UnitGPU::UnitGPU(const char* name_, std::unique_ptr<tflite::Interpreter> interpreter) 
            : name(name_), interpreterGPU(std::move(interpreter)) {
    myThread = std::thread(&UnitGPU::Invoke, this);
}

TfLiteStatus UnitGPU::Invoke() {
    return kTfLiteOk;
}

Interpreter* UnitGPU::GetInterpreter(){return interpreterGPU.get();}

} // End of namespace tflite
