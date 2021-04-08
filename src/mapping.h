#ifndef MAPPING_H
#define MAPPING_H

#ifdef DEBUG
#define DBG(E...) E
#else
#define DBG(E...)
#endif
#ifdef PRINT_TIMES
#include <chrono>
#define TIMES(E...) E
#else
#define TIMES(E...)
#endif

enum interpolation_type
{
	NEAREST_NEIGHBOUR,
	BILINEAR,
	BICUBIC
};

#ifdef CUDA
/**
 * @brief Allocates device memory for the mappings and input and output images.
 * 
 * @param mapx mapping of the x indices
 * @param mapy mapping of the y indices
 * @param width of image in pixel
 * @param height of image in pixel
 */
void init_device_memory(const void *map, unsigned short width, unsigned short height);

/**
 * @brief Frees the device memory.
 */
void free_device_memory();

void cuda_remap_nn(const unsigned char *in, unsigned char *out);
void cuda_remap_li(const unsigned char *in, unsigned char *out);
void cuda_remap_nn(const unsigned char *in_1, const unsigned char *in_2, unsigned char *out);
void cuda_remap_li(const unsigned char *in_1, const unsigned char *in_2, unsigned char *out);
#endif

#endif