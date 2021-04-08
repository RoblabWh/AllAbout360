#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "mapping.h"

#ifdef DEBUG
cudaError_t _cuda_error;
#define CUDA_ERROR_CHECK(FNC_CALL) _cuda_error = FNC_CALL; if(_cuda_error != 0) { printf("%s: %s\n",  #FNC_CALL, cudaGetErrorString(_cuda_error)); exit(EXIT_FAILURE); }
#define CUDA_ERROR_CHECK_KERNEL(KERNEL_CALL...) CUDA_ERROR_CHECK_KERNEL_( (KERNEL_CALL) )
#define CUDA_ERROR_CHECK_KERNEL_(KERNEL_CALL) KERNEL_CALL; _cuda_error = cudaGetLastError(); if(_cuda_error != 0) { printf("%s: %s\n", #KERNEL_CALL, cudaGetErrorString(_cuda_error)); exit(EXIT_FAILURE); }
#else
#define CUDA_ERROR_CHECK(FNC_CALL) FNC_CALL
#define CUDA_ERROR_CHECK_KERNEL(KERNEL_CALL...) KERNEL_CALL
#endif

const unsigned int THREADS_PER_BLOCK = 1024;

double *d_map;
unsigned char *d_idata, *d_odata;
unsigned short h_width, h_height;
unsigned int h_elelen, h_memlen, h_memlen_2;
unsigned short num_blocks;

template<interpolation_type interpol_t>
__global__ void d_remap(unsigned char *g_idata_1, unsigned char *g_idata_2, unsigned char *g_odata, double *g_map, unsigned int len)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len) // fitting number of blocks / threads
    {
        unsigned int i3 = i * 3;
        unsigned int i14 = i * 14;
        unsigned int idx1 = g_map[i14];
        unsigned int idx2 = g_map[i14 + 2];
        double bf1 = g_map[i14 + 4]; // float maybe
        double bf2 = g_map[i14 + 5];
        if (interpol_t == interpolation_type::NEAREST_NEIGHBOUR)
        {
            g_odata[i3] = g_idata_1[idx1] * bf1 + g_idata_2[idx2] * bf2;
            g_odata[i3 + 1] = g_idata_1[idx1 + 1] * bf1 + g_idata_2[idx2 + 1] * bf2;
            g_odata[i3 + 2] = g_idata_1[idx1 + 2] * bf1 + g_idata_2[idx2 + 2] * bf2;
        }
        else if (interpol_t == interpolation_type::BILINEAR)
        {
            unsigned int idx1_y1 = g_map[i14 + 1];
            unsigned int idx2_y1 = g_map[i14 + 3];
            unsigned int idx1_x1 = idx1 + 3;
            unsigned int idx2_x1 = idx2 + 3;
            unsigned int idx1_x1y1 = idx1_y1 + 3;
            unsigned int idx2_x1y1 = idx2_y1 + 3;
            double f11 = g_map[i14 + 6], f12 = g_map[i14 + 7],f13 = g_map[i14 + 8],f14 = g_map[i14 + 9];
            double f21 = g_map[i14 + 10], f22 = g_map[i14 + 11],f23 = g_map[i14 + 12],f24 = g_map[i14 + 13];

            g_odata[i3] = (g_idata_1[idx1] * f11 + g_idata_1[idx1_x1] * f12 + g_idata_1[idx1_y1] * f13 + g_idata_1[idx1_x1y1] * f14) * bf1
                        + (g_idata_2[idx2] * f21 + g_idata_2[idx2_x1] * f22 + g_idata_2[idx2_y1] * f23 + g_idata_2[idx2_x1y1] * f24) * bf2;
            g_odata[i3 + 1] = (g_idata_1[idx1 + 1] * f11 + g_idata_1[idx1_x1 + 1] * f12 + g_idata_1[idx1_y1 + 1] * f13 + g_idata_1[idx1_x1y1 + 1] * f14) * bf1
                            + (g_idata_2[idx2 + 1] * f21 + g_idata_2[idx2_x1 + 1] * f22 + g_idata_2[idx2_y1 + 1] * f23 + g_idata_2[idx2_x1y1 + 1] * f24) * bf2;
            g_odata[i3 + 2] = (g_idata_1[idx1 + 2] * f11 + g_idata_1[idx1_x1 + 2] * f12 + g_idata_1[idx1_y1 + 2] * f13 + g_idata_1[idx1_x1y1 + 2] * f14) * bf1
                            + (g_idata_2[idx2 + 2] * f21 + g_idata_2[idx2_x1 + 2] * f22 + g_idata_2[idx2_y1 + 2] * f23 + g_idata_2[idx2_x1y1 + 2] * f24) * bf2;
        }
    }
}

void init_device_memory(const void *map, unsigned short width, unsigned short height)
{
    h_elelen = width * height;
    h_memlen = h_elelen * 3;
    h_memlen_2 = h_memlen / 2;
    num_blocks = h_elelen % THREADS_PER_BLOCK == 0 ? h_elelen / THREADS_PER_BLOCK : h_elelen / THREADS_PER_BLOCK + 1;
    unsigned int maplen = h_elelen * sizeof(double) * 14;

    DBG( printf("width:%u height:%u elements:%u memory:%u blocks:%u maplen:%u \n", width, height, h_elelen, h_memlen, num_blocks, maplen) );

    CUDA_ERROR_CHECK( cudaMalloc((void **)&d_idata, h_memlen) );
    CUDA_ERROR_CHECK( cudaMalloc((void **)&d_odata, h_memlen) );

    CUDA_ERROR_CHECK( cudaMalloc((void **)&d_map, maplen) );
    CUDA_ERROR_CHECK( cudaMemcpy(d_map, map, maplen, cudaMemcpyHostToDevice) );
}

void free_device_memory()
{
    CUDA_ERROR_CHECK( cudaFree(d_idata) );
    CUDA_ERROR_CHECK( cudaFree(d_odata) );

    CUDA_ERROR_CHECK( cudaFree(d_map) );
}

/**
 * @brief Remaps the given image based on the given mappingfile.
 * 
 * @tparam interpol_t interpolation type
 * @param in Dualfisheye image
 * @param out Equirectangular image
 */
template <interpolation_type interpol_t>
void cuda_remap(const unsigned char *in, unsigned char *out)
{
    CUDA_ERROR_CHECK( cudaMemcpy(d_idata, in, h_memlen, cudaMemcpyHostToDevice) );

    CUDA_ERROR_CHECK_KERNEL( d_remap<interpol_t><<<num_blocks, THREADS_PER_BLOCK>>>(d_idata, d_idata, d_odata, d_map, h_elelen) );

    CUDA_ERROR_CHECK( cudaMemcpy(out, d_odata, h_memlen, cudaMemcpyDeviceToHost) );
}

/**
 * @brief Remaps the given images based on the given mappingfile.
 * 
 * @tparam interpol_t interpolation type
 * @param in_1 first fisheye image
 * @param in_2 second fisheye image
 * @param out Equirectangular image
 */
template <interpolation_type interpol_t>
void cuda_remap(const unsigned char *in_1, const unsigned char *in_2, unsigned char *out)
{
    CUDA_ERROR_CHECK( cudaMemcpy(d_idata, in_1, h_memlen_2, cudaMemcpyHostToDevice) );
    CUDA_ERROR_CHECK( cudaMemcpy(d_idata + h_memlen_2, in_2, h_memlen_2, cudaMemcpyHostToDevice) );

    CUDA_ERROR_CHECK_KERNEL( d_remap<interpol_t><<<num_blocks, THREADS_PER_BLOCK>>>(d_idata, d_idata + h_memlen_2, d_odata, d_map, h_elelen) );

    CUDA_ERROR_CHECK( cudaMemcpy(out, d_odata, h_memlen, cudaMemcpyDeviceToHost) );
}

void cuda_remap_nn(const unsigned char *in, unsigned char *out) { cuda_remap<interpolation_type::NEAREST_NEIGHBOUR>(in, out); }
void cuda_remap_li(const unsigned char *in, unsigned char *out) { cuda_remap<interpolation_type::BILINEAR>(in, out); }
void cuda_remap_nn(const unsigned char *in_1, const unsigned char *in_2, unsigned char *out) { cuda_remap<interpolation_type::NEAREST_NEIGHBOUR>(in_1, in_2, out); }
void cuda_remap_li(const unsigned char *in_1, const unsigned char *in_2, unsigned char *out) { cuda_remap<interpolation_type::BILINEAR>(in_1, in_2, out); }