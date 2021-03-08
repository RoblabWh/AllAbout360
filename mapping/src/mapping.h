#ifndef CUDA_MAPPING_H
#define CUDA_MAPPING_H

#include "vector"

/**
 * @brief Allocates device memory for the mappings and input and output images.
 * 
 * @param mapx mapping of the x indices
 * @param mapy mapping of the y indices
 * @param width of image in pixel
 * @param height of image in pixel
 */
void init_device_memory(const std::vector<unsigned short> &mapx, const std::vector<unsigned short> &mapy,
                       unsigned short width, unsigned short height);

/**
 * @brief Allocates device memory for the mappings and input and output images.
 * 
 * @param mapx mapping of the x indices
 * @param mapy mapping of the y indices
 * @param width of image in pixel
 * @param height of image in pixel
 */
void init_device_memory(const unsigned short *mapx, const unsigned short *mapy, unsigned short width, unsigned short height);


/**
 * @brief Frees the device memory.
 */
void free_device_memory();

/**
 * @brief Maps the given image based on the given mappingfile.
 * 
 * @param in_img Dualfisheye image
 * @param out_img Equirectangular image
 */
void map(const unsigned char *in_img, unsigned char *out_img);

/**
 * @brief Maps the given images based on the given mappingfile.
 * 
 * @param in_img_front front fisheye image
 * @param in_img_rear raer fisheye image
 * @param out_img Equirectangular image
 */
void map(const unsigned char *in_img_front, const unsigned char *in_img_rear, unsigned char *out_img);

#endif