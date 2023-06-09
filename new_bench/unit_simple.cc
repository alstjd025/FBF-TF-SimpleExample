//#include "tensorflow/lite/workframe.h" // legacy header
#include "tensorflow/lite/lite_runtime.h"
#include "tensorflow/lite/util.h"
#include <cmath>
#include <numeric>

#define SEQ 1
#define OUT_SEQ 1
#define mnist 
#define imagenet

#define twomodel

using namespace cv;
using namespace std;

#define RUNTIME_SOCK "/home/nvidia/FBF-TF-SimpleExample/new_bench/sock/runtime_1"
#define SCHEDULER_SOCK "/home/nvidia/FBF-TF-SimpleExample/new_bench/sock/scheduler"

#ifdef mnist

std::vector<std::string> coco_label;
std::vector<std::string> imagenet_label;

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
		for (int i = 0; i < OUT_SEQ; ++i){
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
		for (int i = 0; i < OUT_SEQ; ++i) {
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
	cv::resize(cvimg, cvimg_, cv::Size(416, 416)); //resize
	cvimg_.convertTo(cvimg_, CV_32F, 1.0 / 255.0);


	// Iterate over each pixel in the input image
	// cv::Mat outputImage(cvimg.size(), CV_8UC1);
	// for (int y = 0; y < cvimg.rows; ++y) {
	// 	for (int x = 0; x < cvimg.cols; ++x) {
	// 		// Get the pixel value at the current position
	// 		cv::Vec3b pixel = cvimg.at<cv::Vec3b>(y, x);

	// 		// Calculate the quantized value for each channel (R, G, B)
	// 		unsigned char quantizedValueR = pixel[0] / 255.0 * 255;
	// 		unsigned char quantizedValueG = pixel[1] / 255.0 * 255;
	// 		unsigned char quantizedValueB = pixel[2] / 255.0 * 255;

	// 		// Set the quantized value as the pixel value in the output image
	// 		outputImage.at<unsigned char>(y, x) = (quantizedValueR + quantizedValueG + quantizedValueB) / 3;
	// 	}
	// }

	// input.push_back(outputImage);
	input.push_back(cvimg_);
	//size should be 224 224 for imagenet and mobilenet
	//size should be 416 416 for yolov4, yolov4_tiny
	//size should be 300 300 for ssd-mobilenetv2-lite
}

template<typename dataType>
void softmax(std::vector<dataType>& arr,
															 std::vector<float>& output){
	dataType maxElement = *std::max_element(arr.begin(), arr.end());
	float sum = 0.0;
	for(auto const& i : arr) 
		sum += std::exp(i - maxElement);
	for(int i=0; i<arr.size(); ++i){
		output.push_back(std::exp(arr[i] - maxElement) / sum);
	}
}

template<typename dataType>
void softmax(std::vector<dataType>& arr,
											std::vector<float>& output, int begin){
	arr.erase(arr.begin(), arr.begin()+begin);
	dataType maxElement = *std::max_element(arr.begin(), arr.end());
	float sum = 0.0;
	for(auto const& i : arr) 
		sum += std::exp(i - maxElement);
	for(int i=0; i<arr.size(); ++i){
		output.push_back(std::exp(arr[i] - maxElement) / sum);
	}
}

// Output parser
void PrintRawOutput(std::vector<std::vector<float>*>* output){
	for(int i=0; i< output->size(); ++i){
		printf("CH [%d]\n", i);
		for(int j=0; j<output->at(i)->size(); ++j){
			printf("%.6f ", output->at(i)->at(j));
		}
		printf("\n");
	}
}

float sigmoid(float x) {
  return 1.0 / (1.0 + std::exp(-x));
}

void ParseOutput(std::vector<std::vector<float>*>* output){
	std::vector<float> parsed_output;
	if(output->size() == 1){ // Case of single channel output. (usually classification model)
		for(int i=0; i<output->size(); ++i){
			softmax<float>(*(output->at(i)), parsed_output);
			int max_element = std::max_element(parsed_output.begin(), parsed_output.end()) - parsed_output.begin();
			printf("%d, %.6f \n", max_element, parsed_output[max_element]);
			parsed_output.clear();
		}
		return;
	}
	std::cout << "Got " << output->size() << " outputs to parse" << "\n";
	for(int i=0; i<output->size(); ++i){ // Case of multiple channel output. (which contains bbox, obj score, classification score)
		// parsed_output.push_back(sigmoid(output->at(i)->at(4)));
		parsed_output.push_back(output->at(i)->at(4));
		softmax<float>(*(output->at(i)), parsed_output, 5);
		int max_element = std::max_element(parsed_output.begin()+1, parsed_output.end()) - parsed_output.begin();
		printf("Oscore %.6f, %s %d, %.6f \n", 
				parsed_output[0], coco_label[max_element].c_str(), max_element, parsed_output[max_element]);
		parsed_output.clear();
	}	
	// std::cout << "parsed_outputs : " << parsed_output.size() << "\n";
	// for(int idx=0; idx<parsed_output.size()-1; ++idx){
	// 	printf("%s :  %.6f\n", imagenet_label[idx].c_str(), parsed_output[idx]);
	// }
}

void ParseLabels(){
	std::string coco_file = "coco_label.txt";
	std::string imagenet_file = "imagenet_label.txt";
	std::ifstream coco_fd, imagenet_fd;
	coco_fd.open(coco_file);
	imagenet_fd.open(imagenet_file);
	std::string label;
	while(getline(coco_fd, label)){
		coco_label.push_back(label);
	}
	while(getline(imagenet_fd, label)){
		imagenet_label.push_back(label);
	}	
	// for(int i=0; i<imagenet_label.size(); ++i){
	// 	std::cout << imagenet_label[i] << "\n";
	// }
	std::cout << "COCO labels : " << coco_label.size() << "\n";
	std::cout << "IMAGENET labels : " << imagenet_label.size() << "\n";
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
	read_image_opencv("orange.jpg", input_imagenet);
	// read_image_opencv("orange.jpg", input_imagenet);
	#endif

  double response_time = 0;
  struct timespec begin, end;
  int n = 0;
  #ifdef twomodel
	tflite::TfLiteRuntime runtime(RUNTIME_SOCK, SCHEDULER_SOCK,
																	 first_model, second_model, tflite::INPUT_TYPE::IMAGENET416);
  #endif

	// Output vector
	std::vector<std::vector<float>*>* output;

  while(n < OUT_SEQ){
    std::cout << "invoke : " << n << "\n";
    
    // runtime.FeedInputToModel(first_model, input_mnist[n % 2], tflite::INPUT_TYPE::MNIST);
    runtime.FeedInputToModelDebug(first_model, input_imagenet[0], tflite::INPUT_TYPE::IMAGENET416);

    clock_gettime(CLOCK_MONOTONIC, &begin);

		if(runtime.DebugCoInvoke() != kTfLiteOk){
      std::cout << "Invoke ERROR" << "\n";
      return -1;
    }
		
    clock_gettime(CLOCK_MONOTONIC, &end);
    if(n > 0){ // drop first invoke's data.
      double temp_time = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
      response_time += temp_time;
    }
    n++;
		output = runtime.GetFloatOutputInVector();
		// PrintRawOutput(output);
		ParseLabels();
		ParseOutput(output);
  }
  response_time = response_time / OUT_SEQ;
  printf("Average response time for %d invokes : %.6fs \n", OUT_SEQ, response_time);
}
