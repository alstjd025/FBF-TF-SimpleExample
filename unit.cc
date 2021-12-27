#include "unit.h"

std::mutex mtx_lock;

namespace tflite
{

// UnitCPU
UnitCPU::UnitCPU() : name("NONE"), interpreterCPU(nullptr){}

UnitCPU::UnitCPU(tflite::UnitType eType_, std::unique_ptr<tflite::Interpreter>* interpreter) 
            : eType(eType_), interpreterCPU(std::move(interpreter)) {}

TfLiteStatus UnitCPU::Invoke() { 
    for(int o_loop=0; o_loop<OUT_SEQ; o_loop++){
        //mtx_lock.lock();
        for(int k=0; k<SEQ; k++){
            std::cout << "CPU" << k + o_loop*SEQ << "\n";
            for (int i=0; i<Image_x; i++){
                for (int j=0; j<Image_y; j++){
                    interpreterCPU->get()->typed_input_tensor<float>(0)[i*28 + j] = \
                     ((float)input[k].at<uchar>(i, j)/255.0);            
                }
            } 
            // Run inference
            if(interpreterCPU->get()->Invoke() != kTfLiteOk){
                return kTfLiteError;
            }
        }
    }
    std::cout << "[CPU] Job Done" << "\n"; 
    return kTfLiteOk;
}

Interpreter* UnitCPU::GetInterpreter(){return interpreterCPU->get();}

void UnitCPU::SetInput(std::vector<cv::Mat> input_){
    input = input_;
}

tflite::UnitType UnitCPU::GetUnitType(){
    return eType;
}

// UnitGPU
UnitGPU::UnitGPU() : name("NONE"), interpreterGPU(nullptr){}

UnitGPU::UnitGPU(tflite::UnitType eType_, std::unique_ptr<tflite::Interpreter>* interpreter) 
            : eType(eType_), interpreterGPU(std::move(interpreter)) {}

TfLiteStatus UnitGPU::Invoke() {
    std::cout << "Starting GPU Job" << "\n";
    for(int o_loop=0; o_loop<OUT_SEQ; o_loop++){
        //mtx_lock.lock();
        for(int k=0; k<SEQ; k++){
            std::cout << "GPU" << k + o_loop*SEQ << "\n";
            for (int i=0; i<Image_x; i++){
                for ( int j=0; j<Image_y; j++){
                    interpreterGPU->get()->typed_input_tensor<float>(0)[i*28 + j] = ((float)input[k].at<uchar>(i, j)/255.0);            
                    printf("%.4f ", ((float)input[k].at<uchar>(i, j)/255.0));
                }
                std::cout <<"\n";
            }
            // Run inference
            if(interpreterGPU->get()->Invoke() != kTfLiteOk){
                return kTfLiteError;
            }
        }
        //mtx_lock.unlock();
    }
    std::cout << "[GPU] Job Done" << "\n";
    for (int i =0; i<10; i++){
        printf("%0.4f", interpreterGPU->get()->typed_output_tensor<float>(0)[i] );
    }
    std::cout << "\n";
    return kTfLiteOk;
}

Interpreter* UnitGPU::GetInterpreter(){return interpreterGPU->get();}

void UnitGPU::SetInput(std::vector<cv::Mat> input_){
    input = input_;
}

tflite::UnitType UnitGPU::GetUnitType(){
    return eType;
}

} // End of namespace tflite
