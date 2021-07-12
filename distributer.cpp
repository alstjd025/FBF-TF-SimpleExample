#include "distributer.h"

namespace tflite
{

DistributerCPU::DistributerCPU() : name(nullptr), interpreterCPU(nullptr)
                                    {}


DistributerCPU::DistributerCPU(char* name, Interpreter* interpreter) 
                            : name(name)
{
    interpreterCPU = interpreter;
}

DistributerGPU::DistributerGPU() : name(nullptr), interpreterGPU(nullptr)
                                    {}

DistributerGPU::DistributerGPU(char* name, Interpreter* interpreter) 
                            : name(name)
{
    interpreterGPU = interpreter;
}


} // End of namespace tflite
