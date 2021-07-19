#include "unit.h"

std::mutex mtx_lock;

namespace tflite
{

// UnitCPU
UnitCPU::UnitCPU() : name("NONE"), interpreterCPU(nullptr){}

UnitCPU::UnitCPU(const char* name_, std::unique_ptr<tflite::Interpreter> interpreter) 
            : name(name_), interpreterCPU(std::move(interpreter)) {}


TfLiteStatus UnitCPU::Invoke() {
    double execution_time = 0;  
    for(int o_loop=0; o_loop<OUT_SEQ; o_loop++){
        for(int k=0; k<SEQ; k++){
            std::cout << "CPU" << k + o_loop*SEQ << "\n";
            for (int i=0; i<Image_x; i++){
                for ( int j=0; j<Image_y; j++){
                    std::cout << "adafsdf cpu \n";
                    interpreterCPU->typed_input_tensor<float>(0)[i*28 + j] = ((float)(*input)[k].at<uchar>(i, j)/255.0);            
                }
            } 
            clock_t tStart = clock();
            // Run inference
            TFLITE_MINIMAL_CHECK(interpreterCPU->Invoke() == kTfLiteOk);
            clock_t tEnd = clock();
            clock_t Elepsed = tEnd - tStart;
            execution_time = execution_time + ((double)Elepsed / CLOCKS_PER_SEC);
        }
    }
    execution_time /=(SEQ * OUT_SEQ);
    std::cout << "[CPU] Job Done" << "\n"; 
    cpu_t = execution_time;
    return kTfLiteOk;
}

Interpreter* UnitCPU::GetInterpreter(){return interpreterCPU.get();}

void UnitCPU::SetInput(std::vector<cv::Mat>* input_){
    input = input_;
}




// UnitGPU
UnitGPU::UnitGPU() : name("NONE"), interpreterGPU(nullptr){}

UnitGPU::UnitGPU(const char* name_, std::unique_ptr<tflite::Interpreter> interpreter) 
            : name(name_), interpreterGPU(std::move(interpreter)) {}

TfLiteStatus UnitGPU::Invoke() {
    std::cout << "Starting GPU Job" << "\n";
    double execution_time = 0;
    for(int o_loop=0; o_loop<OUT_SEQ; o_loop++){
        for(int k=0; k<SEQ; k++){
            std::cout << "GPU" << k + o_loop*SEQ << "\n";
            for (int i=0; i<Image_x; i++){
                for ( int j=0; j<Image_y; j++){
                    std::cout << "GPUUUU \n"; 
                    interpreterGPU->typed_input_tensor<float>(0)[i*28 + j] = ((float)(*input)[k].at<uchar>(i, j)/255.0);            
                }
            }
            std::cout << "Adsdad \n";
            clock_t tStart = clock();
            // Run inference
            TFLITE_MINIMAL_CHECK(interpreterGPU->Invoke() == kTfLiteOk);
            clock_t tEnd = clock();
            std::cout << "adadada \n";
            clock_t Elepsed = tEnd - tStart;
            execution_time = execution_time + ((double)Elepsed / CLOCKS_PER_SEC);
        }
    }
    execution_time /=(SEQ * OUT_SEQ);
    std::cout << "[GPU] Job Done" << "\n";
    gpu_t = execution_time;
    return kTfLiteOk;
}

Interpreter* UnitGPU::GetInterpreter(){return interpreterGPU.get();}

void UnitGPU::SetInput(std::vector<cv::Mat>* input_){
    input = input_;
}

} // End of namespace tflite
