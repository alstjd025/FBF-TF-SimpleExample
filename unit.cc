#include "unit.h"

std::mutex mtx_lock;

namespace tflite
{

// UnitCPU
UnitCPU::UnitCPU() : name("NONE"), interpreterCPU(nullptr){}

UnitCPU::UnitCPU(const char* name_, std::unique_ptr<tflite::Interpreter>* interpreter) 
            : name(name_), interpreterCPU(std::move(interpreter)) {}


TfLiteStatus UnitCPU::Invoke() { 
    for(int o_loop=0; o_loop<OUT_SEQ; o_loop++){
        //mtx_lock.lock();
        for(int k=0; k<SEQ; k++){
            std::cout << "CPU" << k + o_loop*SEQ << "\n";
            for (int i=0; i<Image_x; i++){
                for (int j=0; j<Image_y; j++){
                    //std::cout << "xy : " << i*28 + j << " : "<< ((float)input[k].at<uchar>(i, j)/255.0) <<"\n";
                    interpreterCPU->get()->typed_input_tensor<float>(0)[i*28 + j] = \
                     ((float)input[k].at<uchar>(i, j)/255.0);            
                }
            } 
            // Run inference
            TFLITE_MINIMAL_CHECK(interpreterCPU->get()->Invoke() == kTfLiteOk);
        }
        //mtx_lock.unlock();
    }
    std::cout << "[CPU] Job Done" << "\n"; 
    return kTfLiteOk;
}

Interpreter* UnitCPU::GetInterpreter(){return interpreterCPU->get();}

void UnitCPU::SetInput(std::vector<cv::Mat> input_){
    input = input_;
}




// UnitGPU
UnitGPU::UnitGPU() : name("NONE"), interpreterGPU(nullptr){}

UnitGPU::UnitGPU(const char* name_, std::unique_ptr<tflite::Interpreter>* interpreter) 
            : name(name_), interpreterGPU(std::move(interpreter)) {}

TfLiteStatus UnitGPU::Invoke() {
    std::cout << "Starting GPU Job" << "\n";
    for(int o_loop=0; o_loop<OUT_SEQ; o_loop++){
        //mtx_lock.lock();
        for(int k=0; k<SEQ; k++){
            std::cout << "GPU" << k + o_loop*SEQ << "\n";
            for (int i=0; i<Image_x; i++){
                for ( int j=0; j<Image_y; j++){
                    interpreterGPU->get()->typed_input_tensor<float>(0)[i*28 + j] = \
                    ((float)input[k].at<uchar>(i, j)/255.0);            
                }
            }
            // Run inference
            TFLITE_MINIMAL_CHECK(interpreterGPU->get()->Invoke() == kTfLiteOk);
        }
        //mtx_lock.unlock();
    }
    std::cout << "[GPU] Job Done" << "\n";
    return kTfLiteOk;
}

Interpreter* UnitGPU::GetInterpreter(){return interpreterGPU->get();}

void UnitGPU::SetInput(std::vector<cv::Mat> input_){
    input = input_;
}

} // End of namespace tflite
