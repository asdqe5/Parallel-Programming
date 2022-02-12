#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>

typedef unsigned char BYTE;
using namespace std;

void sharpen(cv::Mat &img, cv::Mat &out, int height, int width);
void histogram_equalization(cv::Mat &img, cv::Mat &out, int *hist, int *hist_sum, int height, int width);

void main()
{
	int i, j, y, x;

	cv::Mat img = cv::imread("input.jpg");

	//그레이 스케일 영상으로 변환
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, CV_BGR2GRAY);
	cv::namedWindow("Input");
	cv::imshow("Input", img_gray);

	//이미지 가로 세로
	int width = img_gray.cols;
	int height = img_gray.rows;
	cv::Mat output_hist(height, width, CV_8UC1);
	cv::Mat output_sharp(height, width, CV_8UC1);

	//입력 그레이스케일 영상의 히스토그램 계산
	int *Histogram = new int[256];
	int *Histogram_Sum = new int[256];
	
	clock_t start = clock();
	for (int k = 0; k < 10; k++)
	{
		histogram_equalization(img_gray, output_hist, Histogram, Histogram_Sum, height, width);
		sharpen(img_gray, output_sharp, height, width);
	}
	clock_t end = clock();
	
	//총 실행 시간
	printf("Elapsed time(CPU) = %u ms\n", end - start);

	cv::namedWindow("output_hist");
	cv::imshow("output_hist", output_hist);
	cv::namedWindow("output_sharp");
	cv::imshow("output_sharp", output_sharp);
	cv::imwrite("output_sharp.jpg", output_sharp);
	cv::imwrite("output_hist.jpg", output_hist);

	cv::waitKey(0);
}

void histogram_equalization(cv::Mat &img, cv::Mat &out, int *hist, int *hist_sum, int height, int width)
{
	int sum = 0;
	for (int i = 0; i < 256; i++)
	{
		hist[i] = 0;
		hist_sum[i] = 0;
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int value = img.at<uchar>(i, j);
			hist[value] += 1;
		}
	}

	//입력 영상의 누적 히스토그램 계산
	for (int i = 0; i < 256; i++)
	{
		sum += hist[i];
		hist_sum[i] = sum;
	}

	//입력 그레이스케일 영상의 정규화된 누적 히스토그램 계산
	float normalized_Histogram_Sum[256] = { 0.0, };
	int image_size = height * width;
	for (int i = 0; i < 256; i++)
	{
		normalized_Histogram_Sum[i] = hist_sum[i] / (float)image_size;
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			out.at<uchar>(i, j) = (BYTE)(round(normalized_Histogram_Sum[img.at<uchar>(i, j)] * 255));
		}
	}
}

void sharpen(cv::Mat &img, cv::Mat &out, int height, int width)
{
	cv::Mat Memtemp(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Memtemp.at<uchar>(i,j) = img.at<uchar>(i,j);
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (j >= 0 && j <= width - 1 && i == 0) 
			{
				out.at<uchar>(i,j) = Memtemp.at<uchar>(i, j);
			}
			else if (j >= 0 && j <= width - 1 && i == height - 1) 
			{
				out.at<uchar>(i, j) = Memtemp.at<uchar>(i, j);
			}
			else if (i >= 0 && i <= height - 1 && j == width - 1) 
			{
				out.at<uchar>(i, j) = Memtemp.at<uchar>(i, j);
			}
			else if (i >= 0 && i <= height - 1 && j == 0) 
			{
				out.at<uchar>(i, j) = Memtemp.at<uchar>(i, j);
			}
			else 
			{
				out.at<uchar>(i, j) = 5 * Memtemp.at<uchar>(i, j) - (Memtemp.at<uchar>(i, j - 1) + Memtemp.at<uchar>(i, j + 1) + Memtemp.at<uchar>(i - 1, j) + Memtemp.at<uchar>(i + 1, j));
			}
		}
	}
}