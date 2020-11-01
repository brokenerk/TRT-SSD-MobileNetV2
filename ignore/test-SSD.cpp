// g++ test-SSD.cpp -o test-SSD `pkg-config --cflags --libs opencv4`
#include <opencv2/highgui.hpp>
#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>
#include <iostream>

using namespace std;

int main( int argc, char** argv ) {

	cv::VideoCapture cap("./../camara/videoSSD.mp4");

	if(!cap.isOpened()){
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	while(1){

		cv::Mat image;
		// Capture frame-by-frame
		cap >> image;
		// If the frame is empty, break immediately
		if (image.empty())
			break;

		// Display the resulting frame
		cv::imshow("SSD-Camera-Test", image);
		// Press  ESC on keyboard to exit
		char c = (char)cv::waitKey(25);
		if(c == 27)
			break;
	}

	// When everything done, release the video capture object
	cap.release();

	// Closes all the frames
	cv::destroyAllWindows();
	return 0;
}