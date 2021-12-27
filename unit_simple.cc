#include "tensorflow/lite/unit_handler.h"
#define SEQ 1
#define OUT_SEQ 1000
#define catdog

using namespace cv;
using namespace std;


#ifdef mnist
int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
/*

*/
void read_Mnist(string filename, vector<cv::Mat>& vec) {
	ifstream file(filename, ios::binary);
	if (file.is_open()){
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)& magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)& number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
        file.read((char*)& n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)& n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < SEQ; ++i){
			cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
			for (int r = 0; r < n_rows; ++r){
				for (int c = 0; c < n_cols; ++c){
					unsigned char temp = 0;
					file.read((char*)& temp, sizeof(temp));
					tp.at<uchar>(r, c) = (int)temp;
                    //cout << (float)tp.at<uchar>(r, c) << "\n";
				}
			}
			vec.push_back(tp);
            cout << "Get " << i << " Images" << "\n";
		}
	}
	else {
		cout << "file open failed" << endl;
	}
}

void read_Mnist_Label(string filename, vector<unsigned char> &arr) {
	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		for (int i = 0; i < SEQ; ++i) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			if (i > 7) {
				cout << (int)temp << " ";
				arr.push_back((unsigned char)temp);
			}
		}
	}
	else {
        cout << "file open failed" << endl;
    }
}
#endif

#ifdef catdog
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
#endif

int main(int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "minimal <tflite model>\n");
        return 1;
    }

    const char* filename = argv[1];
    
	vector<cv::Mat> input;
    vector<unsigned char> arr;
	#ifdef mnist
    read_Mnist("train-images-idx3-ubyte", input);
	read_Mnist_Label("train-labels-idx1-ubyte", arr);
	#endif

	#ifdef catdog
	read_image_opencv("cat.0.jpg", input);
	#endif

	tflite::UnitHandler Uhandler(filename);
    
	if (Uhandler.Invoke(UnitType::CPU0, UnitType::GPU0, input) != kTfLiteOk){
		Uhandler.PrintMsg("Invoke Returned Error");
		exit(1);
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
