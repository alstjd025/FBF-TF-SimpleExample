#include "unit_handler.h"

int main(int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "minimal <tflite model>\n");
        return 1;
    }
    const char* filename = argv[1];
    tflite::UnitHandler Uhandler(filename, "input");
    
    if(Uhandler.CreateUnitCPU("CPU1") != kTfLiteOk){
        std::cout << "Cannot Create UnitCPU" << "\n";
        return 1;
    }
    if(Uhandler.CreateUnitGPU("GPU1") != kTfLiteOk){
        std::cout << "Cannot Create UnitCPU" << "\n";
        return 1;
    }
    
    Uhandler.PrintInterpreterStatus();
}