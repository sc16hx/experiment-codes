/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is an example of how to use SeetaFace engine for face detection, the
 * face detection method described in the following paper:
 *
 *
 *   Funnel-structured cascade for multi-view face detection with alignment awareness,
 *   Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
 *   In Neurocomputing (under review)
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Shuzhe Wu (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */
#define _CRT_SECURE_NO_WARNINGS

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <io.h> 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 //#include "opencv2/imgproc/imgproc.hpp"

#include "face_detection.h"
#include "globalmatting.h" 
#include "guidedfilter.h" 
#include "getopt.h" 
#include "bayesian.h" 
#include "sharedmatting.h"

using namespace std;

int detectFace(string imgPath, string detectPath) {
	const char *imgs = imgPath.data();
	const char *detectors = detectPath.data();
	//const char* img_path = imgPath;
	seeta::FaceDetection detector(detectors);

	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	cv::Mat img = cv::imread(imgs, cv::IMREAD_UNCHANGED);
	cv::Mat img_gray;

	if (img.channels() != 1)
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	else
		img_gray = img;

	seeta::ImageData img_data;
	img_data.data = img_gray.data;
	img_data.width = img_gray.cols;
	img_data.height = img_gray.rows;
	img_data.num_channels = 1;

	long t0 = cv::getTickCount();
	std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
	long t1 = cv::getTickCount();
	double secs = (t1 - t0) / cv::getTickFrequency();

	cv::Rect face_rect;
	int32_t num_face = static_cast<int32_t>(faces.size());

	for (int32_t i = 0; i < num_face; i++) {
		face_rect.x = faces[i].bbox.x;
		face_rect.y = faces[i].bbox.y;
		face_rect.width = faces[i].bbox.width;
		face_rect.height = faces[i].bbox.height;

		cv::rectangle(img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);
	}

	if (num_face == 0) {
		return 0;
	}
	else {
		return 1;
	}
}

int resizeFace(string imgPath, string detectPath) {
	int flag = 0;
	string imgName = imgPath;
	imgName.erase(imgName.begin(), imgName.begin() + 14);
	cout << "img name: " <<imgName << endl;
	const char *imgs = imgPath.data(); 	
	const char *detectors = detectPath.data();

	seeta::FaceDetection detector(detectors);
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);
	cv::Mat initImg = cv::imread(imgs, cv::IMREAD_UNCHANGED);
	cv::Mat img = cv::imread(imgs, cv::IMREAD_UNCHANGED);

	cv::Mat imgmini;
	cv::Rect mini;
	mini.x = img.cols*0.1;
	mini.y = img.rows*0.1;
	mini.height = img.rows*0.6;
	mini.width = img.cols*0.6;
	cv::Mat temps = img.clone();

	img = img(mini);

	cv::Mat img_gray;

	if (img.channels() != 1)
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	else
		img_gray = img;

	cv::Mat centerROI, centerGray, cutROI, reImg;
	cv::Rect centerRect, cutRect;

	seeta::ImageData img_data;
	img_data.data = img_gray.data;
	img_data.width = img_gray.cols;
	img_data.height = img_gray.rows;
	img_data.num_channels = 1;

	cv::Rect face_rect;
	std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
	int32_t num_face = static_cast<int32_t>(faces.size());

	if (num_face != 0) {
		flag = 1;
		for (int32_t i = 0; i < 1; i++) {
			face_rect.x = mini.x + faces[i].bbox.x - faces[i].bbox.width*0.4375;
			face_rect.y = mini.y + faces[i].bbox.y - faces[i].bbox.width*0.75;

			face_rect.width = faces[i].bbox.width*1.875;
			face_rect.height = faces[i].bbox.width*2.5;
			cv::Mat result(face_rect.height, face_rect.width, CV_8UC3, cv::Scalar(255, 255, 255));

			for (int i = 0; i < face_rect.width; i++)
			{
				for (int j = 0; j < face_rect.height; j++)
				{
					if (i + face_rect.x < 0 || j + face_rect.y < 0 || i + face_rect.x >= temps.cols || j + face_rect.y >= temps.rows)
					{
						return 0;
						//continue;
					}
					else
					{
						result.at<cv::Vec3b>(j, i) = temps.at<cv::Vec3b>(j + face_rect.y, i + face_rect.x);
					}
				}
			}

			double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
			int y = 0;

			if (face_rect.x < 0)
				x1 = 0 - face_rect.x;
			else
				x1 = 0;
			if (face_rect.width + face_rect.x > temps.cols)
				x2 = temps.cols - face_rect.x;
			else
				x2 = face_rect.width;
			y = (int)((x2 - x1)*4.0 / 3.0);
			int chazhi = temps.rows - y;

			if (face_rect.y < 0)
				y1 = 0 - face_rect.y;
			else
				y1 = 0;
			if (face_rect.height + face_rect.y > temps.rows)
				y2 = temps.rows - face_rect.y;
			else
				y2 = face_rect.height;

			int heng = x2 - x1;
			int zong = y2 - y1;
			cv::Rect xiaotu_rect(x1, y1, heng, zong);

			double zong_2 = (face_rect.width*1.0) / (heng*1.0)*zong;
			double heng_2 = face_rect.width;
			cv::Mat xiaotu = result(xiaotu_rect).clone();
			cv::resize(xiaotu, xiaotu, cv::Size(heng_2, zong_2));
			int tem = (int)zong_2 - face_rect.height;
			if (tem > 0)
			{
				cv::Rect R_A(0, 0, heng_2, face_rect.height);
				xiaotu = xiaotu(R_A).clone();
			}
			if (tem < 0)
			{
				tem = 0 - tem;

				cv::Mat background(face_rect.height, face_rect.width, CV_8UC3, cv::Scalar(0, 255, 0)); //background color
				cv::Rect R_B(0, tem, xiaotu.cols, xiaotu.rows);
				xiaotu.copyTo(background(R_B));
				xiaotu = background.clone();
			}
			cv::Size reSize = cv::Size(300, 400);
			cv::resize(xiaotu, reImg, reSize);
			string savePath = "../test_seeta/resized/" + imgName;
			cv::imwrite(savePath,reImg);
			return 1;
		}
	}
	else
	{
		flag = 0;
		return 0;
	}
}

int otsu(const cv::Mat &img)//otsu algorithm
{
	// calculate histogram
	float histogram[256] = { 0 };
	for (int i = 0; i < img.rows; i++)
	{
		const unsigned char* p = (const unsigned char*)img.ptr(i);
		for (int j = 0; j < img.cols; j++)
		{
			histogram[p[j]]++;
		}
	}
	// average histogram and pixel value: avgValue
	float avgValue = 0;
	int numPixel = img.cols*img.rows;
	for (int i = 0; i < 256; i++)
	{
		histogram[i] = histogram[i] / numPixel;
		avgValue += i * histogram[i];
	}
	// Calculate the maximum variance
	int threshold = 0;
	float gmax = 0, wk = 0, uk = 0;
	for (int i = 0; i < 256; i++) {

		wk += histogram[i];
		uk += i * histogram[i];

		float ut = avgValue * wk - uk;
		float g = ut * ut / (wk*(1 - wk));

		if (g > gmax)
		{
			gmax = g;
			threshold = i;
		}
	}
	return threshold;
}

void segment(string imgPath) {
	const char *imgs = imgPath.data();
	string imgName = imgPath; 	
	imgName.erase(imgName.begin(), imgName.begin() + 14);
	cv::Mat image = cv::imread(imgs, cv::IMREAD_UNCHANGED);
	cv::Mat img_gray;

	if (image.channels() != 1)
		cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
	else
		img_gray = image;

	cv::Rect rectangle(4, 40, image.cols - 5, image.rows - 40);
	cv::Mat result;
	cv::Mat bgModel, fgModel;

	
	result = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(cv::GC_BGD));
	cv::Mat roi(result, rectangle);
	roi = cv::Scalar(cv::GC_PR_FGD);
	//The two steps can be merged (the value of using bgModel and fgModel is reflected here)
	cv::grabCut(image, result, rectangle, bgModel, fgModel, 1,
		cv::GC_INIT_WITH_MASK);
	cv::grabCut(image, result, rectangle, bgModel, fgModel, 4,
		cv::GC_INIT_WITH_MASK);


	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	//result = result & 1 ;
	cv::Mat foreground(image.size(), CV_8UC3,
		cv::Scalar(255, 255, 255));
	//cv::imshow("Grabcut", result);
	image.copyTo(foreground, result);

	//cv::imshow("After", foreground);
	string grabPath = "../test_seeta/grabcut/" + imgName;
	cv::imwrite(grabPath, result);

	//Threshold
	cv::Mat srcGray;
	cvtColor(image, srcGray, CV_BGR2GRAY);
	//imshow("binary", srcGray);
	cv::Mat thresh;
	//otsu
	threshold(srcGray, thresh, 200, 255, CV_THRESH_OTSU);

	//imshow("thresh", thresh);
	//threshold(srcGray, thresh, 0, 255, CV_THRESH_BINARY_INV);

	int gray_threshold = 0;//Threshold segmentation
	gray_threshold = otsu(srcGray);
	printf("gray:" + gray_threshold);
	cv::Mat erzhitu;
	threshold(srcGray, erzhitu, gray_threshold, 255, 1);
	//imshow("threshold", erzhitu);
	string otsuPath = "../test_seeta/otsu/" + imgName;
	cv::imwrite(otsuPath, erzhitu);






	//watershed algorithm

	//1.Use filter2D and Laplace operator to improve image contrast-sharp	
	cv::Mat kernel1 = (cv::Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
	cv::Mat imgLaplance;
	cv::Mat imgSharpen;
	filter2D(image, imgLaplance, CV_32F, kernel1, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
	image.convertTo(imgSharpen, CV_32F);
	cv::Mat imgResult = imgSharpen - imgLaplance;
	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplance.convertTo(imgLaplance, CV_8UC3);
	//imshow("sharpen img", imgResult); 	

	//2. Convert to binary image through threshold 	
	cv::Mat imgBinary;
	cvtColor(imgResult, imgResult, CV_BGR2GRAY);
	threshold(imgResult, imgBinary, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cv::Mat temp;
	imgBinary.copyTo(temp, cv::Mat());
	cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2), cv::Point(-1, -1));
	morphologyEx(temp, temp, CV_MOP_TOPHAT, kernel2, cv::Point(-1, -1), 1);
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			imgBinary.at<uchar>(row, col) = cv::saturate_cast<uchar>(imgBinary.at<uchar>(row, col) - temp.at<uchar>(row, col));
		}
	}
	//imshow("gray-sharpen img", imgResult); 	
	//imshow("binary img", imgBinary); 	

	//3. Distance Transformation 	
	cv::Mat imgDist;
	distanceTransform(imgBinary, imgDist, CV_DIST_L1, 3);

	//4. Morphological operation- the purpose is to remove interference and make the result better 	
	cv::Mat k = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
	morphologyEx(imgDist, imgDist, cv::MORPH_ERODE, k);// Corrosion, interference of de-adhesion site
	//imshow("distance result erode", imgDist);

	//5. Find contours by using findContours
	cv::Mat imgDist8U;
	imgDist.convertTo(imgDist8U, CV_8U);
	vector<vector<cv::Point>> contour;
	findContours(imgDist8U, contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	//6. Draw contours by using drawContours 	
	cv::Mat maskers = cv::Mat::zeros(imgDist8U.size(), CV_32SC1);
	for (size_t i = 0; i < contour.size(); i++) {
		drawContours(maskers, contour, static_cast<int>(i), cv::Scalar::all(static_cast<int>(i) + 1));
	}
	//imshow("maskers", maskers); 	    	

	//7. Watershed transformation
	watershed(image, maskers);
	cv::Mat mark = cv::Mat::zeros(maskers.size(), CV_8UC1);
	maskers.convertTo(mark, CV_8UC1);
	bitwise_not(mark, mark, cv::Mat());
	threshold(mark, mark, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//imshow("watershed", mark);
	string waterPath = "../test_seeta/water/" + imgName;
	cv::imwrite(waterPath, mark);
}

int matting(string imgPath, string trimapPath) {
	const char *imgs = imgPath.data(); 	
	string imgName = imgPath; 	
	imgName.erase(imgName.begin(), imgName.begin() + 22); 	
	cout << "img name: " << imgName << endl;
	const char *trimaps = trimapPath.data();

	cv::Mat image = cv::imread(imgs, cv::IMREAD_UNCHANGED); 	
	cv::Mat img_gray;  	
	if (image.channels() != 1) 		
		cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY); 	
	else 		
		img_gray = image;

	cv::Mat trimap = cv::imread(trimaps, cv::IMREAD_UNCHANGED);
	// bayesian Matting
	BayesianMatting matting(image, trimap);
	cv::Mat bayesResult;
	bayesResult = matting.Solve();
	//imshow("bayes thresh", bayesResult);
	string bayPath = "../test_seeta/bayes/" + imgName;
	bayesResult.convertTo(bayesResult, CV_8UC3, 255);
	cv::imwrite(bayPath, bayesResult);

	
	// shared matting
	char fileAddr[64] = { 0 };
	for (int n = 1; n < 28; ++n) {
		SharedMatting sm;

		sprintf(fileAddr, imgs, n / 10, n % 10);
		sm.loadImage(fileAddr);

		sprintf(fileAddr, trimaps, n / 10, n % 10);
		sm.loadTrimap(fileAddr);

		sm.solveAlpha();

		string sharedPath = "../test_seeta/shared/" + imgName;
		const char *shared = sharedPath.data();
		sprintf(fileAddr, shared, n / 10, n % 10);
		sm.save(fileAddr);
	}
	

	// global Matting
	cv::Mat fore, alpha;
	globalMatting(image, trimap, fore, alpha);

	// filter the result with fast guided filter
	alpha = guidedFilter(image, alpha, 9, 1e-5);
	for (int x = 0; x < trimap.cols; ++x)
		for (int y = 0; y < trimap.rows; ++y)
		{
			if (trimap.at<uchar>(y, x) == 0)
				alpha.at<uchar>(y, x) = 0;
			else if (trimap.at<uchar>(y, x) == 255)
				alpha.at<uchar>(y, x) = 255;
		}
	//imshow("global alpha", alpha);
	string gloPath = "../test_seeta/global/" + imgName; 	
	cv::imwrite(gloPath, alpha);
	//cv::imwrite("../matting_data/global/7.jpg", alpha); //store the resized img
	//image.copyTo(foreground2, alpha);
	//cv::Mat result3(image.size(), CV_8UC3);
	//bitwise_and(image, alpha, result3);
	
}

void getEvalute(cv::Mat src, cv::Mat test) {
	int height = src.rows;//height 	
	int width = src.cols;//width
	int Rs = 0; //Rs, reference area of the segmented image outlined by GT
	int Ts = 0; //Ts, real area of the image segmented by the algorithm
	int Os = 0; //Os.number of pixels that should not be included in the segmentation result
	int Us = 0; //Us,number of pixels that should be included in the segmentation result

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int m = src.at<uchar>(i, j);
			int n = test.at<uchar>(i, j);
			if (m == 255) {
				Rs += 1;
				if (n != 255) {
					Us += 1;
				}
			}
			if (n == 255) {
				Ts += 1;
				if (m != 255) {
					Os += 1;
				}
			}
		}
	}

	double SegRate;       //Segmentation Rate; 
	double OverSegRate;   //Over-segmentation Rate; 
	double LessSegRate;   //Less-segmentation Rate;

	SegRate = 1 - (double)(abs(Rs - Ts)) / Rs; //Segmentation Rate;
	OverSegRate = (double)Os / (Rs + Os); //Over-segmentation Rate; 
	LessSegRate = (double)Us / (Rs + Os); //Less-segmentation Rate;

	//cout << "GroundTruth: " << Rs << endl;
	//cout << "Tested Algorithms: " << Ts << endl;
	cout << "SegRate: " << SegRate << endl;
	cout << "OverSegRate: " << OverSegRate << endl;
	cout << "LessSegRate: " << LessSegRate << endl;
}


cv::Mat h_forpng(cv::Mat picture)
{
	cv::Mat After(picture.rows, picture.cols, CV_8UC3, cv::Scalar(255, 255, 255));//Here is to change the background of the picture and unify the colors.
	int i, j;							    //Of course, you can directly change the background here

	for (i = 0; i < picture.rows; i++)
	{
		for (j = 0; j < picture.cols; j++)
		{
			After.at<cv::Vec3b>(i, j)[0] = picture.at<cv::Vec4b>(i, j)[0] * float(picture.at<cv::Vec4b>(i, j)[3]) / 255 + After.at<cv::Vec3b>(i, j)[0] * float(255 - picture.at<cv::Vec4b>(i, j)[3]) / 255;
			After.at<cv::Vec3b>(i, j)[1] = picture.at<cv::Vec4b>(i, j)[1] * float(picture.at<cv::Vec4b>(i, j)[3]) / 255 + After.at<cv::Vec3b>(i, j)[1] * float(255 - picture.at<cv::Vec4b>(i, j)[3]) / 255;
			After.at<cv::Vec3b>(i, j)[2] = picture.at<cv::Vec4b>(i, j)[2] * float(picture.at<cv::Vec4b>(i, j)[3]) / 255 + After.at<cv::Vec3b>(i, j)[2] * float(255 - picture.at<cv::Vec4b>(i, j)[3]) / 255;
		}
	}
	//Here, if you directly operate on the picture and do not need to store it,  
	//it is recommended to bring it in with a pointer and change it to void to avoid the waste of resources caused by reading in a large picture.
	return After;
}


void genGT(string imgPath) {
	const char *imgs = imgPath.data(); 	
	string imgName = imgPath;
	imgName.erase(imgName.begin(), imgName.begin() + 17);
	cv::Mat image = cv::imread(imgs, cv::IMREAD_UNCHANGED);
	cv::Mat img_gray;

	if (image.channels() != 1)
		cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
	else
		img_gray = image;
	

	cv::Mat srcGray1;
	cv::cvtColor(image, srcGray1, cv::COLOR_BGR2GRAY);
	cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
	imshow("img", image);
	imshow("gray", srcGray1);
	cv::Mat thresh1;
	threshold(srcGray1, thresh1, 254, 255, CV_THRESH_BINARY_INV);
	imshow("bayes thresh", thresh1);
	string savePath = "../test_seeta/groundTruth/" + imgName;
	cout << savePath << endl;
	cv::imwrite(savePath, thresh1); //store the resized img

}

void getTrimap(string imgPath) {
	const char *imgs = imgPath.data(); 	
	string imgName = imgPath;
	imgName.erase(imgName.begin(), imgName.begin() + 22);
	cv::Mat image = cv::imread(imgs, cv::IMREAD_UNCHANGED);
	cv::Mat img_gray;

	if (image.channels() != 1)
		cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
	else
		img_gray = image;

	cv::Rect rectangle(4, 40, image.cols - 5, image.rows - 40);
	cv::Mat result;
	cv::Mat bgModel, fgModel;

	//When the last parameter of grabCut () is cv :: GC_INIT_WITH_MASK
	result = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(cv::GC_BGD));
	cv::Mat roi(result, rectangle);
	roi = cv::Scalar(cv::GC_PR_FGD);
	//The two steps can be merged (the value of using bgModel and fgModel is reflected here)
	cv::grabCut(image, result, rectangle, bgModel, fgModel, 1,
		cv::GC_INIT_WITH_MASK);
	cv::grabCut(image, result, rectangle, bgModel, fgModel, 2,
		cv::GC_INIT_WITH_MASK);


	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	//result = result & 1 ;
	cv::Mat foreground(image.size(), CV_8UC3,
		cv::Scalar(255, 255, 255));
	cv::imshow("Mask", result);
	image.copyTo(foreground, result);
	
	cv::Mat srcGray;
	cvtColor(foreground, srcGray, CV_BGR2GRAY);
	//imshow("srcGray", srcGray);
	cv::Mat thresh;
	threshold(srcGray, thresh, 230, 255, CV_THRESH_BINARY_INV);
	//imshow("thresh", thresh);
	//Custom core
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13));
	//open operation
	cv::Mat open_result;
	cv::morphologyEx(thresh, open_result, cv::MORPH_OPEN, element);
	//imshow("open_result", open_result);
	//close operation
	cv::Mat close_result;
	cv::morphologyEx(thresh, close_result, cv::MORPH_CLOSE, element);
	//imshow("close_result", close_result);
	//Morphological gradient
	//dilate
	cv::Mat dilate_result;
	dilate(close_result, dilate_result, element);
	//imshow("dilate_result", dilate_result);
	//erode
	cv::Mat erode_result;
	erode(close_result, erode_result, element);
	//imshow("erode_result", erode_result);
	cv::Mat outd;
	cv::Mat interme(dilate_result.size(), CV_8UC4);
	//xor to get the gap between erode result and dilate result
	bitwise_xor(dilate_result, erode_result, interme);
	int height = interme.rows;//height
	int width = interme.cols;//width
	//for all pixels
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			//int gray = gray_src.at<uchar>(row, col);
			if (interme.at<uchar>(row, col) > 128)
				interme.at<uchar>(row, col) = 128;
		}
	}
	//imshow("xor", interme);

	cv::Mat gradient_result;
	cv::morphologyEx(close_result, gradient_result, cv::MORPH_GRADIENT, element);
	

	cv::Mat foreground2(dilate_result.size(), CV_8UC3,
		cv::Scalar(255, 255, 255));
	//cv::Mat trimap(foreground.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat out;
	cv::Mat trimap(image.size(), CV_8UC4);
	bitwise_or(interme, erode_result, trimap);
	imshow("trimap", trimap);
	cout << imgName << endl;
	string savePath = "../test_seeta/trimap/" + imgName;
	cv::imwrite(savePath, trimap);
}

void get_image_names(std::string file_path, std::vector<std::string>& file_names)
{
	intptr_t hFile = 0;
	_finddata_t fileInfo;
	hFile = _findfirst(file_path.c_str(), &fileInfo);
	if (hFile != -1) {
		do {
			//If it is a folder, you can continue to traverse through recursion, here we do not need
			if ((fileInfo.attrib &  _A_SUBDIR)) {
				continue;
			}
			//If it is a single file, push_back directly
			else {
				file_names.push_back(fileInfo.name);
				//cout << fileInfo.name << endl;
			}

		} while (_findnext(hFile, &fileInfo) == 0);

		_findclose(hFile);
	}
}

void compareSeg(string gtPath, string imgPath) {
	const char *imgs = imgPath.data();
	const char *gt = gtPath.data();
	string imgName = imgPath;
	cv::Mat image = cv::imread(imgs, cv::IMREAD_UNCHANGED);
	cv::Mat GT = cv::imread(gt, cv::IMREAD_UNCHANGED);
	cv::Mat img_gray;

	if (image.channels() != 1)
		cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
	else
		img_gray = image;

	getEvalute(GT, image);
}

int main() {
	cv::String pattern = "../test_seeta/*.jpg";
	string detectPath = "../FaceDetection/model/seeta_fd_frontal_v1.0.bin";
	//vector<cv::Mat> images = read_images_in_folder(pattern);
	/*
	std::vector<std::string> file_names;
	get_image_names("../test_seeta/*.jpg", file_names);
	cout << file_names.size() << endl;
	int num = 0;
	
	for (size_t i = 0; i < 5; ++i) {
		cout << file_names[i] << endl;
		string img = "../test_seeta/" + file_names[i];
		int face = detectFace(img, detectPath);
		num += face;
	}
	
	cout << "find " << num << " faces in " << file_names.size() << " images." << endl;
	
	for (size_t i = 0; i < 60; ++i) {
		cout << file_names[i] << endl;
		string img = "../test_seeta/" + file_names[i];
		//resizeFace(img, detectPath);
		segment(img);
		//num += face;
	}
	*/
	std::vector<std::string> resize_file_names;
	get_image_names("../test_seeta/resized/*.jpg", resize_file_names);
	for (size_t i = 0; i < 50; ++i) {
		//cout << resize_file_names[i] << endl;
		string img = "../test_seeta/resized/" + resize_file_names[i];
		string tri = "../test_seeta/trimap/" + resize_file_names[i];
		matting(img, tri);
		//getTrimap(img);
	}
	//cout <<"100 images as input, "<< resize_file_names.size() << " images resized successfully." << endl;
	
	/*
	std::vector<std::string> png_file_names; 	
	get_image_names("../test_seeta/groundTruth/*.jpg", png_file_names); 	
	
	for (size_t i = 20; i < 25; ++i) {
		//cout << png_file_names[i] << endl;
		string p = "../test_seeta/groundTruth/" + png_file_names[i];
		//genGT(p);
		string otsuP = "../test_seeta/otsu/" + png_file_names[i];
		compareSeg(p, otsuP);
	}
	cout << png_file_names.size() << endl;
	*/
	cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
