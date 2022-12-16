#include "tensorflow/lite/unit_handler.h"
#define SEQ 1
#define OUT_SEQ 1
#define mnist 

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
	cv::resize(cvimg, cvimg_, cv::Size(300, 300)); //resize to 300x300
	input.push_back(cvimg_);
}


int main(int argc, char* argv[])
{
	const char* originalfilename;
	const char* quantizedfilename;
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

	read_image_opencv("cat.0.jpg", input);
	std::cout << "Loading Cat Image \n";

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
