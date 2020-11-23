#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "mapping.h"

#ifdef DEBUG
#define DBG(E...) E
cudaError_t _cuda_error;
#define CUDA_ERROR_CHECK(FNC_CALL) _cuda_error = FNC_CALL; if(_cuda_error != 0) { printf("%s: %s\n",  #FNC_CALL, cudaGetErrorString(_cuda_error)); exit(EXIT_FAILURE); }
#define CUDA_ERROR_CHECK_KERNEL(KERNEL_CALL...) CUDA_ERROR_CHECK_KERNEL_( (KERNEL_CALL) )
#define CUDA_ERROR_CHECK_KERNEL_(KERNEL_CALL) KERNEL_CALL; _cuda_error = cudaGetLastError(); if(_cuda_error != 0) { printf("%s: %s\n", #KERNEL_CALL, cudaGetErrorString(_cuda_error)); exit(EXIT_FAILURE); }
#else
#define DBG(E...)
#define CUDA_ERROR_CHECK(FNC_CALL) FNC_CALL
#define CUDA_ERROR_CHECK_KERNEL(KERNEL_CALL...) KERNEL_CALL
#endif

const unsigned int THREADS_PER_BLOCK = 1024;

unsigned short *d_mapx, *d_mapy;
unsigned char *d_idata, *d_odata;
unsigned short h_width, h_height;
unsigned int h_elelen, h_memlen, h_memlen_2;
unsigned short num_blocks;

__global__
void d_map(unsigned char *g_idata, unsigned char *g_odata, unsigned short *g_mapx, unsigned short *g_mapy, unsigned int size_row, unsigned int len)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len)
    {
        unsigned int index = (g_mapy[i] * size_row + g_mapx[i]) * 3;
        unsigned int i3 = i * 3;
        g_odata[i3] = g_idata[index];
        g_odata[i3 + 1] = g_idata[index + 1];
        g_odata[i3 + 2] = g_idata[index + 2];
    }
}

__global__
void d_map(unsigned char *g_idataf, unsigned char *g_idatar, unsigned char *g_odata, unsigned short *g_mapx, unsigned short *g_mapy, unsigned int size_row, unsigned int len)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < len)
	{
		unsigned int index;
		unsigned int i3 = i * 3;
		if (g_mapx[i] < size_row)
		{
			index = (g_mapy[i] * size_row + g_mapx[i]) * 3;
			g_odata[i3] = g_idataf[index];
			g_odata[i3 + 1] = g_idataf[index + 1];
			g_odata[i3 + 2] = g_idataf[index + 2];
		}
		else
		{
			index = (g_mapy[i] * size_row + g_mapx[i] - size_row) * 3;
			g_odata[i3] = g_idatar[index];
			g_odata[i3 + 1] = g_idatar[index + 1];
			g_odata[i3 + 2] = g_idatar[index + 2];
		}
	}
}

void init_device_memory(const std::vector<unsigned short>& mapx, const std::vector<unsigned short>& mapy, 
    unsigned short width, unsigned short height)
{
    init_device_memory(mapx.data(), mapy.data(), width, height);
}

void init_device_memory(const unsigned short *mapx, const unsigned short *mapy, unsigned short width, unsigned short height)
{
    h_width = width;
    h_height = height;
    h_elelen = width * height;
    h_memlen = h_elelen * 3;
    h_memlen_2 = h_memlen / 2;
    num_blocks = h_elelen % THREADS_PER_BLOCK == 0 ? h_elelen / THREADS_PER_BLOCK : h_elelen / THREADS_PER_BLOCK + 1;
    unsigned int maplen = h_elelen * sizeof(unsigned short);

    DBG( printf("width:%u height:%u elements:%u memory:%u blocks:%u \n", h_width, h_height, h_elelen, h_memlen, num_blocks) );

    CUDA_ERROR_CHECK( cudaMalloc((void **)&d_idata, h_memlen) );
    CUDA_ERROR_CHECK( cudaMalloc((void **)&d_odata, h_memlen) );

    CUDA_ERROR_CHECK( cudaMalloc((void **)&d_mapx, maplen) );
    CUDA_ERROR_CHECK( cudaMalloc((void **)&d_mapy, maplen) );

    CUDA_ERROR_CHECK( cudaMemcpy(d_mapx, mapx, maplen, cudaMemcpyHostToDevice) );
    CUDA_ERROR_CHECK( cudaMemcpy(d_mapy, mapy, maplen, cudaMemcpyHostToDevice) );
}

void free_device_memory()
{
    CUDA_ERROR_CHECK( cudaFree(d_idata) );
    CUDA_ERROR_CHECK( cudaFree(d_odata) );

    CUDA_ERROR_CHECK( cudaFree(d_mapx) );
    CUDA_ERROR_CHECK( cudaFree(d_mapy) );
}


void map(const unsigned char *in_img, unsigned char *out_img)
{
    CUDA_ERROR_CHECK( cudaMemcpy(d_idata, in_img, h_memlen, cudaMemcpyHostToDevice) );
    
    CUDA_ERROR_CHECK_KERNEL( d_map<<<num_blocks, THREADS_PER_BLOCK>>>(d_idata, d_odata, d_mapx, d_mapy, h_width, h_elelen) );

    CUDA_ERROR_CHECK( cudaMemcpy(out_img, d_odata, h_memlen, cudaMemcpyDeviceToHost) );
}

void map(const unsigned char *in_img_front, const unsigned char *in_img_rear, unsigned char *out_img)
{
    CUDA_ERROR_CHECK( cudaMemcpy(d_idata, in_img_front, h_memlen_2, cudaMemcpyHostToDevice) );
    CUDA_ERROR_CHECK( cudaMemcpy(d_idata + h_memlen_2, in_img_rear, h_memlen_2, cudaMemcpyHostToDevice) );

    CUDA_ERROR_CHECK_KERNEL( d_map<<<num_blocks, THREADS_PER_BLOCK>>>(d_idata, d_idata + h_memlen_2, d_odata, d_mapx, d_mapy, h_height, h_elelen) );

    CUDA_ERROR_CHECK( cudaMemcpy(out_img, d_odata, h_memlen, cudaMemcpyDeviceToHost) );
}