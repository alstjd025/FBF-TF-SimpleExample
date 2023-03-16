#include "tensorflow/lite/unit_handler.h"
#define SEQ 1
#define OUT_SEQ 1

using namespace cv;
using namespace std;



void read_image_opencv(string filename, vector<cv::Mat>& input){
	cv::Mat cvimg = cv::imread(filename, cv::IMREAD_COLOR);
	if(cvimg.data == NULL){
		std::cout << "=== IMAGE DATA NULL ===\n";
		return;
	}
	cv::cvtColor(cvimg, cvimg, COLOR_BGR2RGB);
	cv::Mat cvimg_;
	cv::resize(cvimg, cvimg_, cv::Size(416, 416), 0, 0, INTER_AREA); //resize to 224x224
	cvimg_.convertTo(cvimg_, CV_32FC3, 1 / 127.5f, -1);
	input.push_back(cvimg_);
	//size should be 224 224 for imagenet and mobilenet
	//size should be 416 416 for yolov4
	//size should be 300 300 for ssd-mobilenetv2-lite
}


int main(int argc, char* argv[])
{
	const char* originalfilename;
	const char* quantizedfilename;
  //const char* imagefilename;
	bool bUseTwoModel = false;
	if (argc == 2) {
		std::cout << "Got One Model \n";
		originalfilename = argv[1];
	}
	else if(argc > 2){
		std::cout << "Got Two Model \n";
		bUseTwoModel = true;
		originalfilename = argv[1];
		quantizedfilename = argv[2];
	}
	else{
			fprintf(stderr, "minimal <tflite model>\n");
			return 1;
	}
	vector<cv::Mat> input;
	vector<unsigned char> arr;

	std::cout << "Loading imagenet Image \n";
	read_image_opencv("/home/nvidia/FBF-TF-SimpleExample/mobilenet/apple.jpg", input);

	if(!bUseTwoModel){
		tflite::UnitHandler Uhandler(originalfilename);
		if (Uhandler.Invoke(UnitType::CPU0, UnitType::GPU0, input) != kTfLiteOk){
			Uhandler.PrintMsg("Invoke Returned Error");
			exit(1);
		}		
	}
	else{
		tflite::UnitHandler Uhandler(originalfilename, quantizedfilename);
		if (Uhandler.Invoke(UnitType::CPU0, UnitType::GPU0, input) != kTfLiteOk){
			Uhandler.PrintMsg("Invoke Returned Error");
			exit(1);
		}
	}
	/*
    if(Uhandler.CreateUnitCPU(tflite::UnitType::CPU0, vCPU) != kTfLiteOk){
        std::cout << "Cannot Create UnitCPU" << "\n";
        return 1;
    }
	
    if(Uhandler.CreateUnitGPU(tflite::UnitType::GPU0, vGPU) != kTfLiteOk){
        std::cout << "Cannot Create UnitGPU" << "\n";
        return 1;
    }
    
    
    if(Uhandler.Invoke() != kTfLiteOk){
        std::cout << "Invoke Error" << "\n";
        return 1;
    }
	*/
}
