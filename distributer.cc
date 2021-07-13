#include "distributer.h"

namespace tflite
{

DistributerCPU::DistributerCPU() : name(nullptr), interpreterCPU(nullptr)
                                    {}


DistributerCPU::DistributerCPU(const char* name, Interpreter* interpreter) 
                            : name(name)
{
    interpreterCPU = interpreter;
}

Interpreter* DistributerCPU::GetInterpreter(){return interpreterCPU;}

DistributerGPU::DistributerGPU() : name(nullptr), interpreterGPU(nullptr)
                                    {}

DistributerGPU::DistributerGPU(const char* name, Interpreter* interpreter) 
                            : name(name)
{
    interpreterGPU = interpreter;
}

Interpreter* DistributerGPU::GetInterpreter(){return interpreterGPU;}

} // End of namespace tflite
