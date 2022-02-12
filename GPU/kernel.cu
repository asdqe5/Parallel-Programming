#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>

typedef unsigned char BYTE;
using namespace std;

cv::Mat convertToMat(unsigned char *buffer, int width, int height);

__global__ void histogram_equalization(BYTE *img, BYTE *out, int *hist, int *hist_sum, int height, int width);
__global__ void histogram_equalization2(BYTE *img, BYTE *out, int *hist, float *hist_sum, int height, int width);
__global__ void sharpen(BYTE *img, BYTE *out, BYTE *temp, int height, int width);

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
	cv::Mat output1(height, width, CV_8UC1);
	cv::Mat output2(height, width, CV_8UC1);
	BYTE *img_buffer = new BYTE[height * width];
	BYTE *out_buffer1 = new BYTE[height * width];
	BYTE *out_buffer2 = new BYTE[height * width];
	
	//Mat 을 배열로 전환
	uchar* p;
	for (int j = 0; j < height; j++)
	{
		p = img_gray.ptr<uchar>(j);
		for (int i = 0; i < width; i++)
		{
			img_buffer[j * width + i] = p[i];
		}
	}

	//입력 그레이스케일 영상의 히스토그램 계산
	float normalized_Histogram_Sum[256] = { 0.0, };
	int image_size = height * width;

	int *Histogram = new int[256];
	int *Histogram_Sum = new int[256];

	//GPU 설정
	BYTE *dev_buffer1;
	BYTE *dev_buffer2;
	BYTE *dev_img;
	BYTE *Memtemp = new BYTE[height * width];
	BYTE *dev_temp;

	int *dev_Histogram = new int[256];
	int *dev_Histogram_Sum = new int[256];
	float *dev_normal = new float[256];

	cudaSetDevice(0);

	dim3 dimGrid((width - 1) / 32 + 1, (height - 1) / 32 + 1);
	dim3 dimBlock(32, 32);

	//히스토그램 평탄화 시작
	clock_t start = clock();
	for (int k = 0; k < 10; k++)
	{
		int sum = 0;

		for (int i = 0; i < 256; i++)
		{
			Histogram[i] = 0;
			Histogram_Sum[i] = 0;
		}

		cudaMalloc((void **)&dev_img, sizeof(BYTE) * height * width);
		cudaMalloc((void **)&dev_buffer1, sizeof(BYTE) * height *width);
		cudaMalloc((void **)&dev_buffer2, sizeof(BYTE) * height *width);
		cudaMalloc((void **)&dev_Histogram, sizeof(int) * 256);
		cudaMalloc((void **)&dev_Histogram_Sum, sizeof(int) * 256);
		cudaMalloc((void **)&dev_normal, sizeof(float) * 256);
		cudaMalloc((void **)&dev_temp, sizeof(BYTE) * height * width);

		cudaMemcpy(dev_img, img_buffer, sizeof(BYTE) * height * width, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_buffer1, out_buffer1, sizeof(BYTE) * height *width, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_buffer2, out_buffer2, sizeof(BYTE) * height *width, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_temp, Memtemp, sizeof(BYTE) * height *width, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Histogram, Histogram, sizeof(int) * 256, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Histogram_Sum, Histogram_Sum, sizeof(int) * 256, cudaMemcpyHostToDevice);
		histogram_equalization << <dimGrid, dimBlock >> > (dev_img, dev_buffer1, dev_Histogram, dev_Histogram_Sum, height, width);

		cudaMemcpy(Histogram, dev_Histogram, sizeof(int) * 256, cudaMemcpyDeviceToHost);
		cudaMemcpy(Histogram_Sum, dev_Histogram_Sum, sizeof(int) * 256, cudaMemcpyDeviceToHost);

		//입력 영상의 누적 히스토그램 계산
		for (int i = 0; i < 256; i++)
		{
			sum += Histogram[i];
			Histogram_Sum[i] = sum;
		}

		//입력 그레이스케일 영상의 정규화된 누적 히스토그램 계산
		for (int i = 0; i < 256; i++)
		{
			normalized_Histogram_Sum[i] = Histogram_Sum[i] / (float)image_size;
		}

		cudaMemcpy(dev_normal, normalized_Histogram_Sum, sizeof(float) * 256, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Histogram_Sum, Histogram_Sum, sizeof(int) * 256, cudaMemcpyHostToDevice);

		histogram_equalization2 << <dimGrid, dimBlock >> > (dev_img, dev_buffer1, dev_Histogram, dev_normal, height, width);

		cudaMemcpy(out_buffer1, dev_buffer1, sizeof(BYTE) * width * height, cudaMemcpyDeviceToHost);

		//sharpening
		sharpen << <dimGrid, dimBlock >> > (dev_img, dev_buffer2, dev_temp, height, width);
		cudaDeviceSynchronize();

		cudaMemcpy(out_buffer2, dev_buffer2, sizeof(BYTE) * width * height, cudaMemcpyDeviceToHost);
	}
	clock_t end = clock();

	output1 = convertToMat(out_buffer1, width, height);
	output2 = convertToMat(out_buffer2, width, height);

	//총 실행 시간
	printf("Elapsed time(GPU) = %u ms\n", end - start);

	cv::namedWindow("output_hist");
	cv::imshow("output_hist", output1);
	cv::namedWindow("output_sharp");
	cv::imshow("output_sharp", output2);
	cv::imwrite("output_sharp.jpg", output1);
	cv::imwrite("output_hist.jpg", output2);

	cudaDeviceReset();

	cudaFree(dev_img);
	cudaFree(dev_buffer1);
	cudaFree(dev_buffer2);
	cudaFree(dev_Histogram);
	cudaFree(dev_Histogram_Sum);
	cudaFree(dev_normal);
	cudaFree(dev_temp);

	cv::waitKey(0);
}

__global__ void histogram_equalization2(BYTE *img, BYTE *out, int *hist, float *hist_sum, int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ((i >= height) || (j >= width))
		return;

	out[i * width + j] = (BYTE)(round(hist_sum[img[i * width + j]] * 255));
	__syncthreads();
}

cv::Mat convertToMat(unsigned char *buffer, int width, int height)
{
	cv::Mat tmp(height, width, CV_8UC1);

	for (int x = 0; x < height; x++) 
	{
		for (int y = 0; y < width; y++)
		{
			tmp.at<uchar>(x, y) = buffer[x * width + y];
		}
	}
	return tmp;
}

__global__ void histogram_equalization(BYTE *img, BYTE *out, int *hist, int *hist_sum, int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ((i >= height) || (j >= width))
		return;
	int value = img[i * width + j];
	atomicAdd(&hist[value], 1);
	__syncthreads();
}

__global__ void sharpen(BYTE *img, BYTE *out, BYTE *temp, int height, int width)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i >= height) || (j >= width))
		return;

	temp[i * width + j] = img[i * width + j];
	__syncthreads();

	if (j >= 0 && j <= width - 1 && i == 0) 
	{
		out[i*width + j] = temp[i*width + j];
	}
	else if (j >= 0 && j <= width - 1 && i == height - 1) 
	{
		out[i*width + j] = temp[i*width + j];
	}
	else if (i >= 0 && i <= height - 1 && j == width - 1) 
	{
		out[i*width + j] = temp[i*width + j];
	}
	else if (i >= 0 && i <= height - 1 && j == 0) 
	{
		out[i*width + j] = temp[i*width + j];
	}
	else 
	{
		out[i*width + j] = 5 * temp[i*width + j] - (temp[i*width + j - 1] + temp[i*width + j + 1] + temp[(i - 1)*width + j] + temp[(i + 1)*width + j]);
	}
	__syncthreads();
}