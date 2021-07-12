#include "distributer.h"
#include "distributer_handler.h"

int main(int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "minimal <tflite model>\n");
        return 1;
    }
    const char* filename = argv[1];
    tflite::DistributerHandler Dis(filename, "input");
    if(Dis.CreateDistributerCPU("CPU1") != kTfLiteOk){
        std::cout << "Cannot Create DistributerCPU" << "\n";
        return 1;
    }
    if(Dis.CreateDistributerGPU("GPU1") != kTfLiteOk){
        std::cout << "Cannot Create DistributerCPU" << "\n";
        return 1;
    }

    Dis.PrintInterpreterStatus();
}