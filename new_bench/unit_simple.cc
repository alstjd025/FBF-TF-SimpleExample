//#include "tensorflow/lite/workframe.h" // legacy header
#include "tensorflow/lite/lite_runtime.h"
#include "tensorflow/lite/util.h"
#define SEQ 1
#define OUT_SEQ 2
#define mnist 
#define imagenet

#define twomodel

using namespace cv;
using namespace std;

#define RUNTIME_SOCK "/home/nvidia/FBF-TF-SimpleExample/new_bench/sock/runtime_1"
#define SCHEDULER_SOCK "/home/nvidia/FBF-TF-SimpleExample/new_bench/sock/scheduler"

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
                    // cout << (float)tp.at<uchar>(r, c) << "\n";
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

void read_image_opencv(string filename, vector<cv::Mat>& input){
	cv::Mat cvimg = cv::imread(filename, cv::IMREAD_COLOR);
	if(cvimg.data == NULL){
		std::cout << "=== IMAGE DATA NULL ===\n";
		return;
	}
	cv::cvtColor(cvimg, cvimg, COLOR_BGR2RGB);
	cv::Mat cvimg_;
	cv::resize(cvimg, cvimg_, cv::Size(224, 224), 0, 0, INTER_AREA); //resize to 224x224
	cvimg_.convertTo(cvimg_, CV_32FC3, 1 / 127.5f, -1);
	input.push_back(cvimg_);
	//size should be 224 224 for imagenet and mobilenet
	//size should be 416 416 for yolov4
	//size should be 300 300 for ssd-mobilenetv2-lite
}

int main(int argc, char* argv[])
{
	const char* first_model;
	const char* second_model;
	bool bUseTwoModel = false;
	if (argc == 2) {
		std::cout << "Got One Model \n";
		first_model = argv[1];
	}
	else if(argc > 2){
		std::cout << "Got Two Model \n";
		bUseTwoModel = true;
		first_model = argv[1];
		second_model = argv[2];
	}
	else{
			fprintf(stderr, "minimal <tflite model>\n");
			return 1;
	}
	vector<cv::Mat> input_mnist;
	vector<cv::Mat> input_imagenet;
	vector<unsigned char> arr;

	#ifdef mnist
	std::cout << "Loading images \n";
	read_Mnist("train-images-idx3-ubyte", input_mnist);
	std::cout << "Loading Labels \n";
	read_Mnist_Label("train-labels-idx1-ubyte", arr);
	std::cout << "Loading Mnist Image, Label Complete \n";
	#endif

	#ifdef imagenet
	read_image_opencv("banana_0.jpg", input_imagenet);
	read_image_opencv("orange.jpg", input_imagenet);
	#endif

  double response_time = 0;
  struct timespec begin, end;
  int n = 0;
  #ifdef twomodel
	tflite::TfLiteRuntime runtime(RUNTIME_SOCK, SCHEDULER_SOCK,
																	 first_model, second_model, tflite::INPUT_TYPE::MNIST);
  #endif

  #ifndef twomodel
	tflite::TfLiteRuntime runtime(RUNTIME_SOCK, SCHEDULER_SOCK,
																	 first_model, tflite::INPUT_TYPE::IMAGENET224);
  #endif

  while(n < OUT_SEQ){
    std::cout << "invoke : " << n << "\n";
    
    //runtime.FeedInputToModel(first_model, input_imagenet[n % 2], tflite::INPUT_TYPE::MNIST);
    runtime.FeedInputToModelDebug(first_model, input_mnist[n % 2], tflite::INPUT_TYPE::MNIST);

    clock_gettime(CLOCK_MONOTONIC, &begin);
    // if(runtime.Invoke() != kTfLiteOk){
    //   std::cout << "Invoke ERROR" << "\n";
    //   return -1;
    // }
		if(runtime.DebugInvoke() != kTfLiteOk){
      std::cout << "Invoke ERROR" << "\n";
      return -1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    if(n > 0){ // drop first invoke's data.
      double temp_time = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
      response_time += temp_time;
    }
    n++;
  }
  response_time = response_time / OUT_SEQ;
  printf("Average response time for %d invokes : %.6fs \n", OUT_SEQ, response_time);
}
