#include "distributer.h"

namespace tflite
{

// DistributerCPU

DistributerCPU::DistributerCPU() : name("NONE"), interpreterCPU(nullptr){}

DistributerCPU::DistributerCPU(char* name_, std::unique_ptr<tflite::Interpreter> interpreter) 
{
    std::strcpy(name, name_);
    interpreterCPU = std::move(interpreter);
}

Interpreter* DistributerCPU::GetInterpreter(){return interpreterCPU.get();}

// DistributerGPU

DistributerGPU::DistributerGPU() : name("NONE"), interpreterGPU(nullptr){}

DistributerGPU::DistributerGPU(char* name_, std::unique_ptr<tflite::Interpreter> interpreter) 
    
{
    std::strcpy(name, name_);
    interpreterGPU = std::move(interpreter);
}

Interpreter* DistributerGPU::GetInterpreter(){return interpreterGPU.get();}

} // End of namespace tflite
