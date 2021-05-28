#ifndef MAPPING_H
#define MAPPING_H

#include <string>
#include <thread>
#include <mutex>
#include <atomic>

#include <opencv2/opencv.hpp>

#ifdef WITH_OPENCL
#include <CL/cl2.hpp>
#endif

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

#ifdef WITH_CUDA
struct extra_data
{
	
};
#else
#ifdef WITH_OPENCL
struct extra_data
{
	static const size_t LOCAL_SIZE = 1024;
	cl::Context ctxt;
	cl::CommandQueue cmdq;
	cl::Program prog;
	cl::Kernel k;
	cl::Buffer map;
	cl::Buffer in_1;
	cl::Buffer in_2;
	cl::Buffer out;
	cl::NDRange local_size = cl::NDRange(LOCAL_SIZE);
	cl::NDRange global_size;
};
#else
struct extra_data {};
#endif
#endif

class mapper //TODO CUDA
{
private:
	cv::VideoCapture cap_1, cap_2;
	cv::Mat in_1, in_2, map, *out;
	std::thread worker;
	std::mutex cap_lock;
	std::atomic_bool *unread, *unwritten;
	size_t read_pos, write_pos, buffer_length, frameskip;
	std::atomic_bool running, alive, good;
	extra_data edata;

	static const size_t DEFAULT_BUFFER_LENGTH = 1;

	template <bool single_input> void init(interpolation_type interpol_t, size_t buffer_length);
	template <interpolation_type interpol_t, bool single_input> void process();

public:
	mapper(cv::VideoCapture &in, const cv::Mat map, size_t frameskip = 0, interpolation_type interpol_t = interpolation_type::NEAREST_NEIGHBOUR, size_t buffer_length = mapper::DEFAULT_BUFFER_LENGTH);
	mapper(cv::VideoCapture &in_1, cv::VideoCapture in_2, const cv::Mat map, size_t frameskip = 0, interpolation_type interpol_t = interpolation_type::NEAREST_NEIGHBOUR, size_t buffer_length = mapper::DEFAULT_BUFFER_LENGTH);
	mapper(const std::string &in_path, const std::string &map_path, size_t frameskip = 0, interpolation_type interpol_t = interpolation_type::NEAREST_NEIGHBOUR, size_t buffer_length = mapper::DEFAULT_BUFFER_LENGTH);
	mapper(const std::string &in_path_1, const std::string &in_path_2, const std::string &map_path, size_t frameskip = 0, interpolation_type interpol_t = interpolation_type::NEAREST_NEIGHBOUR, size_t buffer_length = mapper::DEFAULT_BUFFER_LENGTH);
	~mapper();

	/**
	 * @brief Get the next image. WARNING! img.data is only valid until next call of this function!
	 * 
	 * @param img Output image
	 * @return true Got image successfully.
	 * @return false Could not get new image.
	 */
	bool get_next_img(cv::Mat &img);
	/**
	 * @brief Get the next image as copy.
	 * 
	 * @param img Output image
	 * @return true Got image successfully.
	 * @return false Could not get new image.
	 */
	bool copy_next_img(cv::Mat &img);
	/**
	 * @brief Jumps if input supports it. WARNING! jumps are relative to the worker thread position and the already done buffered images are not cleared!
	 * 
	 * @param ms Milliseconds to jump.
	 * @return true Jumping is supported.
	 * @return false Jumping is not supported.
	 */
	bool jump_mseconds(int ms);
	/**
	 * @brief Jumps if input supports it. WARNING! jumps are relative to the worker thread position and the already done buffered images are not cleared!
	 * 
	 * @param frames Frmaes to jump.
	 * @return true Jumping is supported.
	 * @return false Jumping is not supported.
	 */
	bool jump_frames(int frames);
	/**
	 * @brief Set the frameskip
	 * 
	 * @param frames Number of frames to skip.
	 */
	void set_frameskip(size_t frames);
	/**
	 * @brief Get the frameskip
	 * 
	 * @return size_t Number of frames to skip.
	 */
	size_t get_frameskip();
	/**
	 * @brief Sets the worker thread to running.
	 */
	void start();
	/**
	 * @brief Sets the worker thread to paused.
	 */
	void stop();
};

#ifdef WITH_CUDA
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