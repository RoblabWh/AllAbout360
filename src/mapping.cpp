#include <iostream>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "mapping.h"
#ifdef WITH_OPENCL
#include <CL/cl.hpp>
#include "clerror.h"
#include "mapping.cl.h"
#endif

using namespace std;

const string WINDOW_NAME = "Output";

enum program_mode
{
	INTERACTIVE,
	VIDEO,
	IMAGE
};

struct program_args
{
	string in_path_1, in_path_2, map_path, out_path, out_extension, param_path;
	cv::InterpolationFlags interpol_t = cv::INTER_NEAREST;
	int out_codec = 0, out_dec_len = 0, out_width = 0, out_height = 0;
	double search_x = 0, search_y = 0, search_width = 1, search_height = 1;
	size_t frameskip = 0, num_frames = 0, search_range = 0;
	program_mode prog_mode = program_mode::INTERACTIVE;
	bool preview = false, video_index = false;
	calib_params parameter = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

/**
 * @brief Reads the mapping table from file at path
 *
 * @tparam interpol_t interpolation type to buld the mapping table for
 * @param path path to mapping file
 * @param mapping_table mapping table in format [x1, y1, x2, y2, blend factor 1, blend factor 2]
 * 						or [x1, y1, x2, y2, blend factor 1, blend factor 2, x1 y1 factor, x1+1 y1 factor, x1 y1+1 factor, x1+1 y1+1 factor, x2 y2 factor, x2+1 y2 factor, x2 y2+1 factor, x2+1 y2+1 factor]
 */
template <cv::InterpolationFlags interpol_t, bool single_input>
void read_mapping_file(string path, cv::Mat &mapping_table)
{
	int width, height;
	ifstream file(path);
	if (!file.is_open())
	{
		cerr << "Couldn't open \"" << path << "\" for reading." << endl;
		exit(EXIT_FAILURE);
	}
	if (!(file >> width >> height))
	{
		cerr << "Couldn't read resolution from file \"" << path << "\"." << endl;
		exit(EXIT_FAILURE);
	}
	DBG(cout << "mappingtable size: " << width << 'x' << height << endl;)
	mapping_table.create(height, width, CV_64FC(14)); //TODO size based on interpolation

	cv::Vec<double, 14> mte;
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			if (file >> mte[0] >> mte[1] >> mte[2] >> mte[3] >> mte[4])
			{
				if (mte[0] < 0 || mte[0] > height - 1 || mte[1] < 0 || mte[1] > height - 1 || mte[2] < 0 || mte[2] > height - 1 || mte[3] < 0 || mte[3] > height - 1 || mte[4] < 0 || mte[4] > 1)
				{
					cerr << "Error reading mapping file \"" << path << "\" on line " << j * width + i + 2 << ", value out of range." << endl;
					exit(EXIT_FAILURE);
				}
				mte[5] = 1 - mte[4];
				if constexpr (interpol_t == cv::INTER_LINEAR)
				{
					int ix = mte[0], iy = mte[1];
					double xf = mte[0] - ix, yf = mte[1] - iy;
					double xfi = 1 - xf, yfi = 1 - yf;
					mte[6] = xfi * yfi;
					mte[7] = xf * yfi;
					mte[8] = xfi * yf;
					mte[9] = xf * yf;
					ix = mte[2];
					iy = mte[3];
					xf = mte[2] - ix;
					yf = mte[3] - iy;
					xfi = 1 - xf;
					yfi = 1 - yf;
					mte[10] = xfi * yfi;
					mte[11] = xf * yfi;
					mte[12] = xfi * yf;
					mte[13] = xf * yf;
				}
				else if constexpr (interpol_t == cv::INTER_NEAREST)
				{
					mte[0] = round(mte[0]);
					mte[1] = round(mte[1]);
					mte[2] = round(mte[2]);
					mte[3] = round(mte[3]);
				}
				#ifdef ON_GPU
				if constexpr (single_input)
				{
					mte[0] = (width * ((int) mte[1]) + ((int) mte[0])) * 3;
					mte[1] = mte[0] + width * 3;
					mte[2] = (width * ((int) mte[3]) + ((int) mte[2]) + height) * 3;
					mte[3] = mte[2] + width * 3;
				}
				else
				{
					mte[0] = (height * ((int) mte[1]) + ((int) mte[0])) * 3;
					mte[1] = mte[0] + height * 3;
					mte[2] = (height * ((int) mte[3]) + ((int) mte[2])) * 3;
					mte[3] = mte[2] + height * 3;
				}
				#endif
				mapping_table.at<cv::Vec<double, 14>>(j, i) = mte;
			}
			else
			{
				cerr << "Error reading mapping file \"" << path << "\" on line " << j * width + i + 2 << '.' << endl;
				exit(EXIT_FAILURE);
			}
		}
	}
	if (file >> mte[0])
	{
		cerr << "Error reading mapping file \"" << path << "\" on line " << height * width + 1 << ", too many entries." << endl;
		exit(EXIT_FAILURE);
	}
	file.close();
}

void read_mapping_params(string path, calib_params &params)
{
	ifstream file(path);
	if (!file.is_open())
	{
		cerr << "Couldn't open \"" << path << "\" for reading." << endl;
		exit(EXIT_FAILURE);
	}
	file >> params.x_off_front >> params.y_off_front >> params.radius_front >> params.rot_front
		 >> params.x_off_rear >> params.y_off_rear >> params.radius_rear >> params.rot_rear
		 >> params.front_limit >> params.blend_limit >> params.blend_size >> params.orientation >> params.time_offset;
	if (file.fail())
	{
		cerr << "Wasn't able to read all parameters from file \"" << path << "\"." << endl;
		exit(EXIT_FAILURE);
	}
	char endc;
	file >> endc;
	if (!file.eof())
	{
		cerr << "Parameterfile \"" << path << "\" has too many entries." << endl;
		exit(EXIT_FAILURE);
	}
	file.close();
	DBG(
	cout << "Read parameters:"
		 << "\n  front:"
		 << "\n    x offset:  " << params.x_off_front
		 << "\n    y offset:  " << params.y_off_front
		 << "\n    radius:    " << params.radius_front
		 << "\n    rotation:  " << params.rot_front
		 << "\n  rear:"
		 << "\n    x offset:  " << params.x_off_rear
		 << "\n    y offset:  " << params.y_off_rear
		 << "\n    radius:    " << params.radius_rear
		 << "\n    rotation:  " << params.rot_rear
		 << "\n  front limit: " << params.front_limit
		 << "\n  blend limit: " << params.blend_limit
		 << "\n  blend size:  " << params.blend_size
		 << "\n  time offset: " << params.time_offset
		 << "\n";
	)
}

/**
 * @brief Checks if x, y are in range of height and limits them if not.
 * @param x
 * @param y
 * @param width
 * @param height
 */
void clip_borders(double &x, double &y, unsigned int width, unsigned int height, unsigned char cutoff = 1)
{
	if (x < 0)
		x = 0;
	else if (x > width - cutoff)
		x = width - cutoff;
	if (y < 0)
		y = 0;
	else if (y > height - cutoff)
		y = height - cutoff;
}

/**
 * @brief Calculates the x and y indices and the bland factor for the dualfisheye images.
 *
 * @param p_x 3D normalised cartesian vector x part
 * @param p_y 3D normalised cartesian vector y part
 * @param p_z 3D normalised cartesian vector z part
 * @param width number of cols in input image
 * @param height number of rows in input image
 * @param params calibration parameters
 * @return Vec5d Indices for both fisheye images and the blend factor
 */
template<cv::InterpolationFlags interpol_t, bool single_input, bool expand, typename T>
T cart3D_idxDFE(double p_x, double p_y, double p_z, int width, int height, const calib_params *const params)
{
	// polar normalised fisheye coordinates
	// a = angle from +x or -x axis
	// r = radius on fish eye to pixel
	// t = angle from +y or -y on fish eye to pixel
	double a, r, t;
	double bf;   // blend factor
	double x1, y1, x2, y2; // fisheye index
	T mte; // mapping table entry
	int size = width > height ? width : height;

	unsigned char cutoff;
	switch (interpol_t)
	{
	case cv::INTER_NEAREST:
		cutoff = 1;
		break;
	case cv::	INTER_LINEAR:
		cutoff = 2;
		break;
	default:
		break;
	}

	a = atan2(sqrt(p_y * p_y + p_z * p_z), p_x);

	if (a < params->front_limit) // front
	{
		r = a / M_PI_2 * params->radius_front;
		t = atan2(p_z, p_y) + params->rot_front;

		x1 = (r * cos(t) * size + width) / 2 + params->x_off_front;
		y1 = (r * sin(t) * size + height) / 2 + params->y_off_front;
		clip_borders(x1, y1, width, height, cutoff);
		x2 = 0;
		y2 = 0;
		bf = 1;
	}
	else if (a < params->blend_limit) // blend
	{
		// front
		r = a / M_PI_2 * params->radius_front;
		t = atan2(p_z, p_y) + params->rot_front;

		x1 = (r * cos(t) * size + width) / 2 + params->x_off_front;
		y1 = (r * sin(t) * size + height) / 2 + params->y_off_front;
		clip_borders(x1, y1, width, height, cutoff);

		bf = 1 - ((a - params->front_limit) / params->blend_size);

		// rear
		r = (M_PI - a) / M_PI_2 * params->radius_rear;
		t = atan2(p_z, -p_y) + params->rot_rear;

		x2 = (r * cos(t) * size + width) / 2 + params->x_off_rear;
		y2 = (r * sin(t) * size + height) / 2 + params->y_off_rear;
		clip_borders(x2, y2, width, height, cutoff);
	}
	else // rear
	{
		r = (M_PI - a) / M_PI_2 * params->radius_rear;
		t = atan2(p_z, -p_y) + params->rot_rear;

		x2 = (r * cos(t) * size + width) / 2 + params->x_off_rear;
		y2 = (r * sin(t) * size + height) / 2 + params->y_off_rear;
		clip_borders(x2, y2, width, height, cutoff);

		x1 = 0;
		y1 = 0;

		bf = 0;
	}

	// expand mapping table to reduce calculation later on
	if constexpr (expand)
	{
		double bfi = 1 - bf;
		int xi1 = x1, yi1 = y1, xi2 = x2, yi2 = y2;
		int i1, i2;

		if constexpr (interpol_t == cv::INTER_NEAREST)
		{
			xi1 = round(x1);
			yi1 = round(y1);
			xi2 = round(x2);
			yi2 = round(y2);

			mte[2] = bf;
			mte[3] = bfi;
		}
		else if constexpr (interpol_t == cv::INTER_LINEAR)
		{
			double xf = x1 - xi1, yf = y1 - yi1;
			double xfi = 1 - xf, yfi = 1 - yf;

			mte[2] = xfi * yfi * bf;
			mte[3] = xf * yfi * bf;
			mte[4] = xfi * yf * bf;
			mte[5] = xf * yf * bf;

			xf = x2 - xi2;
			yf = y2 - yi2;
			xfi = 1 - xf;
			yfi = 1 - yf;

			mte[6] = xfi * yfi * bfi;
			mte[7] = xf * yfi * bfi;
			mte[8] = xfi * yf * bfi;
			mte[9] = xf * yf * bfi;
		}

		if constexpr (single_input)
		{
			i1 = width * 2 * yi1 + xi1;
			i2 = width * 2 * yi2 + xi2 + width;
		}
		else
		{
			i1 = width * yi1 + xi1;
			i2 = width * yi2 + xi2;
		}

		#ifdef ON_GPU
		*((int*)&(mte[0])) = i1 * 3;
		*((int*)&(mte[1])) = i2 * 3;
		#else
		mte[0] = i1;
		mte[1] = i2;
		#endif
	}
	return mte;
}

template<cv::InterpolationFlags interpol_t, bool single_input, bool expand>
void gen_equi_mapping_table(int width, int height, int in_width, int in_height, const calib_params *const params, cv::Mat &mapping_table)
{
	int i, j;			  // equirectangular index
	double phi, theta;	  // equirectangular polar and azimuthal angle
	double p_x, p_y, p_z; // 3D normalised cartesian fisheye vector

	double width_2 = width / 2;
	double ratio = width_2 > height ? (double) height  / (double) width_2 : (double) width_2 / (double) height;
	double ratio_width = width_2 > height ? 1 : ratio, ratio_height = width_2 > height ? ratio : 1;
	double ratio_width_offset = M_PI * (1 - ratio_width), ratio_height_offset = M_PI * (1 - ratio_height) / 2;

	#ifdef ON_GPU
	if constexpr (interpol_t == cv::INTER_NEAREST)
		mapping_table.create(height, width, CV_32FC(4));
	else if constexpr (interpol_t == cv::INTER_LINEAR)
		mapping_table.create(height, width, CV_32FC(10));
	#else
	if constexpr (interpol_t == cv::INTER_NEAREST)
		mapping_table.create(height, width, CV_64FC(4));
	else if constexpr (interpol_t == cv::INTER_LINEAR)
		mapping_table.create(height, width, CV_64FC(10));
	#endif

	#pragma omp parallel for private(i, j, phi, theta, p_x, p_y, p_z)
	for (j = 0; j < height; j++)
	{
		theta = (1.0 - ((double)j / (double)height)) * M_PI * ratio_height + ratio_height_offset;
		for (i = 0; i < width; i++)
		{
			phi = (double)i / (double)width * 2.0 * M_PI * ratio_width + ratio_width_offset - params->orientation;

			p_x = cos(phi) * sin(theta);
			p_y = sin(phi) * sin(theta);
			p_z = cos(theta);

			#ifdef ON_GPU
			if constexpr (interpol_t == cv::INTER_NEAREST)
				mapping_table.at<cv::Vec<float, 4>>(j, i) = cart3D_idxDFE<interpol_t, single_input, expand, cv::Vec<float, 4>>(p_x, p_y, p_z, in_width, in_height, params);
			else if constexpr (interpol_t == cv::INTER_LINEAR)
				mapping_table.at<cv::Vec<float, 10>>(j, i) = cart3D_idxDFE<interpol_t, single_input, expand, cv::Vec<float, 10>>(p_x, p_y, p_z, in_width, in_height, params);
			#else
			if constexpr (interpol_t == cv::INTER_NEAREST)
				mapping_table.at<cv::Vec<double, 4>>(j, i) = cart3D_idxDFE<interpol_t, single_input, expand, cv::Vec<double, 4>>(p_x, p_y, p_z, in_width, in_height, params);
			else if constexpr (interpol_t == cv::INTER_LINEAR)
				mapping_table.at<cv::Vec<double, 10>>(j, i) = cart3D_idxDFE<interpol_t, single_input, expand, cv::Vec<double, 10>>(p_x, p_y, p_z, in_width, in_height, params);
			#endif
		}
	}
}

void gen_equi_mapping_table(int width, int height, int in_width, int in_height, cv::InterpolationFlags interpol_t, bool single_input, bool expand, const calib_params *const params, cv::Mat &mapping_table)
{
	if (single_input)
	{
		if (expand)
		{
			switch (interpol_t)
			{
			case cv::INTER_NEAREST:
				gen_equi_mapping_table<cv::INTER_NEAREST, true, true>(width, height, in_width, in_height, params, mapping_table);
				break;
			case cv::INTER_LINEAR:
				gen_equi_mapping_table<cv::INTER_LINEAR, true, true>(width, height, in_width, in_height, params, mapping_table);
				break;
			}
		}
		else
		{
			switch (interpol_t)
			{
			case cv::INTER_NEAREST:
				gen_equi_mapping_table<cv::INTER_NEAREST, true, false>(width, height, in_width, in_height, params, mapping_table);
				break;
			case cv::INTER_LINEAR:
				gen_equi_mapping_table<cv::INTER_LINEAR, true, false>(width, height, in_width, in_height, params, mapping_table);
				break;
			}
		}
	}
	else
	{
		if (expand)
		{
			switch (interpol_t)
			{
			case cv::INTER_NEAREST:
				gen_equi_mapping_table<cv::INTER_NEAREST, false, true>(width, height, in_width, in_height, params, mapping_table);
				break;
			case cv::INTER_LINEAR:
				gen_equi_mapping_table<cv::INTER_LINEAR, false, true>(width, height, in_width, in_height, params, mapping_table);
				break;
			}
		}
		else
		{
			switch (interpol_t)
			{
			case cv::INTER_NEAREST:
				gen_equi_mapping_table<cv::INTER_NEAREST, false, false>(width, height, in_width, in_height, params, mapping_table);
				break;
			case cv::INTER_LINEAR:
				gen_equi_mapping_table<cv::INTER_LINEAR, false, false>(width, height, in_width, in_height, params, mapping_table);
				break;
			}
		}
	}
}

template<cv::InterpolationFlags interpol_t, bool single_input, bool expand>
void gen_cube_mapping_table(int width, int height, int in_height, const calib_params *const params, cv::Mat &mapping_table)
{
	int i, j;			  // equirectangular index
	double p_x, p_y, p_z; // 3D normalised cartesian fisheye vector
	int sqr = width / 3,  // square width/height
		sqr_x2 = sqr * 2,
		sqr_2 = sqr / 2;

	// DBG(cout << "mappingtable size: " << width << 'x' << height << endl;)
	#ifdef ON_GPU
	if constexpr (interpol_t == cv::INTER_NEAREST)
		mapping_table.create(height, width, CV_32FC(4));
	else if constexpr (interpol_t == cv::INTER_LINEAR)
		mapping_table.create(height, width, CV_32FC(10));
	#else
	if constexpr (interpol_t == cv::INTER_NEAREST)
		mapping_table.create(height, width, CV_64FC(4));
	else if constexpr (interpol_t == cv::INTER_LINEAR)
		mapping_table.create(height, width, CV_64FC(10));
	#endif

	#pragma omp parallel for private(i, j, p_x, p_y, p_z) collapse(2)
	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			if (j < sqr)
			{
				if (i < sqr)
				{
					p_x = i - sqr_2;
					p_y = sqr_2;
					p_z = j - sqr_2;
				}
				else if (i < sqr_x2)
				{
					p_x = sqr_2;
					p_y = -((i - sqr) - sqr_2);
					p_z = j - sqr_2;
				}
				else
				{
					p_x = -((i - sqr_x2) - sqr_2);
					p_y = -sqr_2;
					p_z = j - sqr_2;
				}
			}
			else
			{
				if (i < sqr)
				{
					p_x = -sqr_2;
					p_y = i - sqr_2;
					p_z = (j - sqr) - sqr_2;
				}
				else if (i < sqr_x2)
				{
					p_x = -((j - sqr) - sqr_2);
					p_y = -((i - sqr) - sqr_2);
					p_z = sqr_2;
				}
				else
				{
					p_x = (j - sqr) - sqr_2;
					p_y = -((i - sqr_x2) - sqr_2);
					p_z = -sqr_2;
				}
			}
			#ifdef ON_GPU
			if constexpr (interpol_t == cv::INTER_NEAREST)
				mapping_table.at<cv::Vec<float, 4>>(j, i) = cart3D_idxDFE<interpol_t, single_input, expand, cv::Vec<float, 4>>(p_x, p_y, p_z, in_height, params);
			else if constexpr (interpol_t == cv::INTER_LINEAR)
				mapping_table.at<cv::Vec<float, 10>>(j, i) = cart3D_idxDFE<interpol_t, single_input, expand, cv::Vec<float, 10>>(p_x, p_y, p_z, in_height, params);
			#else
			if constexpr (interpol_t == cv::INTER_NEAREST)
				mapping_table.at<cv::Vec<double, 4>>(j, i) = cart3D_idxDFE<interpol_t, single_input, expand, cv::Vec<double, 4>>(p_x, p_y, p_z, in_height, params);
			else if constexpr (interpol_t == cv::INTER_LINEAR)
				mapping_table.at<cv::Vec<double, 10>>(j, i) = cart3D_idxDFE<interpol_t, single_input, expand, cv::Vec<double, 10>>(p_x, p_y, p_z, in_height, params);
			#endif
		}
	}
}

#ifdef WITH_OPENCL
void cl_init_device(cv::InterpolationFlags interpol_t, bool single_input, const cv::Mat &mapping_table, size_t in_width, size_t in_height, const string &source, extra_data *data)
{
	CL_ARGERR_CHECK( data->ctxt = cl::Context(CL_DEVICE_TYPE_GPU, NULL, NULL, NULL, &_cl_error) );
	CL_ARGERR_CHECK( data->cmdq = cl::CommandQueue(data->ctxt, 0, &_cl_error) );
	CL_ARGERR_CHECK( data->prog = cl::Program(data->ctxt, source, true, &_cl_error) );
	size_t elelen = mapping_table.rows * mapping_table.cols, in_memlen = in_width * in_height * 3;
	data->global_size = cl::NDRange(elelen % extra_data::LOCAL_SIZE == 0 ? elelen : elelen + extra_data::LOCAL_SIZE - elelen % extra_data::LOCAL_SIZE);
	CL_ARGERR_CHECK( data->map = cl::Buffer(data->ctxt, CL_MEM_READ_WRITE, mapping_table.dataend - mapping_table.datastart, NULL, &_cl_error) );
	if (single_input)
	{
		CL_ARGERR_CHECK( data->in_1 = cl::Buffer(data->ctxt, CL_MEM_READ_WRITE, in_memlen * 2, NULL, &_cl_error) );
	}
	else
	{
		CL_ARGERR_CHECK( data->in_1 = cl::Buffer(data->ctxt, CL_MEM_READ_WRITE, in_memlen, NULL, &_cl_error) );
		CL_ARGERR_CHECK( data->in_2 = cl::Buffer(data->ctxt, CL_MEM_READ_WRITE, in_memlen, NULL, &_cl_error) );
	}
	CL_ARGERR_CHECK( data->out = cl::Buffer(data->ctxt, CL_MEM_READ_WRITE, elelen * 3, NULL, &_cl_error) );
	switch (interpol_t)
	{
	case cv::INTER_NEAREST:
		CL_ARGERR_CHECK( data->k = cl::Kernel(data->prog, "remap_nearest", &_cl_error) );
		break;
	case cv::INTER_LINEAR:
		CL_ARGERR_CHECK( data->k = cl::Kernel(data->prog, "remap_linear", &_cl_error) );
		if (single_input)
		{
			CL_RETERR_CHECK( data->k.setArg(5, (unsigned int) in_width * 6) );
		}
		else
		{
			CL_RETERR_CHECK( data->k.setArg(5, (unsigned int) in_width * 3) );
		}
		break;
	default:
		cerr << "Interpolation type not supported." << endl;
		exit(EXIT_FAILURE);
		break;
	}
	CL_RETERR_CHECK( data->k.setArg(0, data->in_1) );
	if (single_input)
	{
		CL_RETERR_CHECK( data->k.setArg(1, data->in_1) );
	}
	else
	{
		CL_RETERR_CHECK( data->k.setArg(1, data->in_2) );
	}
	CL_RETERR_CHECK( data->k.setArg(2, data->out) );
	CL_RETERR_CHECK( data->k.setArg(3, data->map) );
	CL_RETERR_CHECK( data->k.setArg(4, (unsigned int) elelen) );
	CL_RETERR_CHECK( data->cmdq.enqueueWriteBuffer(data->map, CL_TRUE, 0, mapping_table.dataend - mapping_table.datastart, mapping_table.data) );
}
#endif

/**
 * @brief  Parses the arguments given in commandline
 *
 * @param argc argc from main
 * @param argv argv from main
 * @param args parsed arguments
 */
void parse_args(int argc, char **argv, program_args &args)
{
	const string USAGE_TEXT = string("usage:\n  ")
							 + argv[0] + string(" -h\n  ")
							 + argv[0] + string(" MODE [OPTION ...] -m <map file> <input file 1> [<input file 2>]\n");
	const string HELP_TEXT = "modes:\n"
							 "  interactive             int  interactively jump through the video and save images\n"
							 "  video                   vid  convert the input to a video file\n"
							 "  image                   img  convert the input to an image sequence\n"
							 "\n"
							 "global options:\n"
							 "  --help                  -h                    print this message\n"
							 "  --mappingfile  --map    -m  PATH              mapping file path\n"
							 "  --parameters  --params  -p  PARAMETERS        calibration parameters\n"
							 "  --parameter-file        -pf PATH              calibration parameter file\n"
							 "  --output                -o  PATH              output file path with printf format for image numbers\n"
							 "  --resolution            -r  WIDTH HEIGHT      output resolution\n"
							 "  --linear-interpolation  -li                   enable linear interpolation\n"
							 "  --search-range          -ra RANGE             range to search befor and after the current frame for sharper image\n"
							 "  --search-region         -re X Y WIDTH HEIGHT  region to use in the current frame for image sharpness detection given as double[x y width height] which are factors in relation to max\n"
							 "video options:\n"
							 "  --codec  --fourcc       -c  CODEC             codec for video output\n"
							 "  --frameskip             -fs NUMBER            number of frames to skip between each image\n"
							 "  --number-frames         -nf NUMBER            number of frames to output with even spacing from start to end\n"
							 "  --preview               -pv                   show images on screen while mapping\n"
							 "image options:\n"
							 "  --frameskip             -fs NUMBER            number of frames to skip between each image\n"
							 "  --number-frames         -nf NUMBER            number of frames to output with even spacing from start to end\n"
							 "  --preview               -pv                   show images on screen while mapping\n"
							 "  --video-index           -vi                   use video index and not output index for indexing output\n";

	// parse mode of operation
	if (argc < 2)
	{
		cerr << "No arguments provided.\n" << USAGE_TEXT << '\n' << HELP_TEXT << endl;
		exit(EXIT_FAILURE);
	}
	if (string("--help") == argv[1] || string("-h") == argv[1])
	{
		cout << USAGE_TEXT << '\n' << HELP_TEXT << endl;
		exit(EXIT_SUCCESS);
	}
	else if (string("video") == argv[1] || string("vid") == argv[1])
		args.prog_mode = program_mode::VIDEO;
	else if (string("image") == argv[1] || string("img") == argv[1])
		args.prog_mode = program_mode::IMAGE;
	else if (string("interactive") == argv[1] || string("int") == argv[1])
		args.prog_mode = program_mode::INTERACTIVE;
	else
	{
		cerr << '"' << argv[1] << "\" is not a valid mode.\n" << USAGE_TEXT << '\n' << HELP_TEXT << endl;
		exit(EXIT_FAILURE);
	}

	// parse args
	int i, inc = 0;
	for (i = 2; i < argc; i++)
	{
		if (string("--output") == argv[i] || string("-o") == argv[i])
		{
			i++;
			if (i < argc)
			{
				args.out_path = argv[i];
				size_t ext_begin = args.out_path.find_last_of('.');
				if (ext_begin == string::npos)
				{
					cerr << "output path \"" << argv[i] << "\" has no filename extension." << endl;
					exit(EXIT_FAILURE);
				}
				args.out_extension = args.out_path.substr(ext_begin);
				args.out_path = args.out_path.substr(0, args.out_path.size() - args.out_extension.size());
				size_t placeholder_begin = args.out_path.find('%'), placeholder_end = args.out_path.find('d', placeholder_begin);
				if (placeholder_begin != string::npos)
				{
					if (placeholder_end == string::npos)
					{
						cerr << "started placeholder '%' at index " << placeholder_begin << " without matching ending 'd'." << endl;
						exit(EXIT_FAILURE);
					}
					args.out_dec_len = stoi(args.out_path.substr(placeholder_begin + 1, placeholder_end - placeholder_begin - 1));
				}
			}
			else
			{
				cerr << "no output path provided." << endl;
				exit(EXIT_FAILURE);
			}
		}
		else if (string("--linear-interpolation") == argv[i] || string("-li") == argv[i])
		{
			args.interpol_t = cv::INTER_LINEAR;
		}
		else if (string("-c") == argv[i] || string("--codec") == argv[i] || string("--fourcc") == argv[i])
		{
			if (args.prog_mode != program_mode::VIDEO)
			{
				cerr << "the codec option is only available in video mode." << endl;
				exit(EXIT_FAILURE);
			}
			i++;
			if (i < argc)
			{
				if (string(argv[i]).length() == 4)
				{
					args.out_codec = cv::VideoWriter::fourcc(argv[i][0], argv[i][1], argv[i][2], argv[i][3]);
				}
				else
				{
					cerr << "codec has to be exectly four characters long." << endl;
					exit(EXIT_FAILURE);
				}
			}
			else
			{
				cerr << "no codec provided." << endl;
				exit(EXIT_FAILURE);
			}
		}
		else if (string("--map") == argv[i] || string("--mappingfile") == argv[i] || string("-m") == argv[i])
		{
			if (!args.param_path.empty() || args.parameter.radius_front != 0)
			{
				cerr << "Parameters and mapping table can't be used together." << endl;
				exit(EXIT_FAILURE);
			}
			i++;
			if (i < argc)
			{
				args.map_path = argv[i];
			}
			else
			{
				cerr << "no map provided." << endl;
				exit(EXIT_FAILURE);
			}
		}
		else if (string("--params") == argv[i] || string("--parameters") == argv[i] || string("-p") == argv[i])
		{
			if (!args.map_path.empty())
			{
				cerr << "Parameters and mapping table can't be used together." << endl;
				exit(EXIT_FAILURE);
			}
			if (!args.param_path.empty())
			{
				cerr << "Parameters can be either given by command line oder by file but not both." << endl;
				exit(EXIT_FAILURE);
			}
			i += 13;
			if (i < argc)
			{
				args.parameter.x_off_front = stod(argv[i - 12]);
				args.parameter.y_off_front = stod(argv[i - 11]);
				args.parameter.radius_front = stod(argv[i - 10]);
				args.parameter.rot_front = stod(argv[i - 9]);
				args.parameter.x_off_rear = stod(argv[i - 8]);
				args.parameter.y_off_rear = stod(argv[i - 7]);
				args.parameter.radius_rear = stod(argv[i - 6]);
				args.parameter.rot_rear = stod(argv[i - 5]);
				args.parameter.front_limit = stod(argv[i - 4]);
				args.parameter.blend_limit = stod(argv[i - 3]);
				args.parameter.blend_size = stod(argv[i - 2]);
				args.parameter.orientation = stod(argv[i - 1]);
				args.parameter.time_offset = stod(argv[i]);
			}
			else
			{
				cerr << "not enough parameters provided." << endl;
				exit(EXIT_FAILURE);
			}
		}
		else if (string("--parameter-file") == argv[i] || string("-pf") == argv[i])
		{
			if (!args.map_path.empty())
			{
				cerr << "Parameters and mapping table can't be used together." << endl;
				exit(EXIT_FAILURE);
			}
			if (args.parameter.radius_front != 0)
			{
				cerr << "Parameters can be either given by command line oder by file but not both." << endl;
				exit(EXIT_FAILURE);
			}
			i++;
			if (i < argc)
			{
				args.param_path = argv[i];
			}
			else
			{
				cerr << "no parameter file provided." << endl;
				exit(EXIT_FAILURE);
			}
		}
		else if (string("-fs") == argv[i] || string("--frameskip") == argv[i])
		{
			if (args.prog_mode != program_mode::VIDEO && args.prog_mode != program_mode::IMAGE)
			{
				cerr << "the frameskip option is only available in video and image mode." << endl;
				exit(EXIT_FAILURE);
			}
			if (args.num_frames != 0)
			{
				cerr << "Only frameskip or number-frames can be set at the same time." << endl;
				exit(EXIT_FAILURE);
			}
			i++;
			if (i < argc)
			{
				args.frameskip = stoul(argv[i]);
			}
			else
			{
				cerr << "no frameskip provided." << endl;
				exit(EXIT_FAILURE);
			}
		}
		else if (string("-nf") == argv[i] || string("--number-frames") == argv[i])
		{
			if (args.prog_mode != program_mode::VIDEO && args.prog_mode != program_mode::IMAGE)
			{
				cerr << "the number frames option is only available in video and image mode." << endl;
				exit(EXIT_FAILURE);
			}
			if (args.frameskip != 0)
			{
				cerr << "Only frameskip or number-frames can be set at the same time." << endl;
				exit(EXIT_FAILURE);
			}
			i++;
			if (i < argc)
			{
				args.num_frames = stoul(argv[i]);
			}
			else
			{
				cerr << "no frameskip provided." << endl;
				exit(EXIT_FAILURE);
			}
		}
		else if (string("-ra") == argv[i] || string("--search-range") == argv[i])
		{
			i++;
			if (i < argc)
			{
				args.search_range = stoul(argv[i]);
			}
			else
			{
				cerr << "no search range provided." << endl;
				exit(EXIT_FAILURE);
			}
		}
		else if (string("-re") == argv[i] || string("--search-region") == argv[i])
		{
			i += 4;
			if (i < argc)
			{
				args.search_x = stod(argv[i - 3]);
				args.search_y = stod(argv[i - 2]);
				args.search_width = stod(argv[i - 1]);
				args.search_height = stod(argv[i]);
			}
			else
			{
				cerr << "search region not or not completely provided." << endl;
				exit(EXIT_FAILURE);
			}
		}
		else if (string("--preview") == argv[i] || string("-pv") == argv[i])
		{
			args.preview = true;
		}
		else if (string("--video-index") == argv[i] || string("-vi") == argv[i])
		{
			if (args.prog_mode != program_mode::IMAGE)
			{
				cerr << "Output indexing is only available in image mode." << endl;
				exit(EXIT_FAILURE);
			}
			args.video_index = true;
		}
		else if (string("--resolution") == argv[i] || string("-r") == argv[i])
		{
			i++;
			if (i < argc)
			{
				args.out_width = stoi(argv[i]);
			}
			else
			{
				cerr << "no resolution provided." << endl;
				exit(EXIT_FAILURE);
			}
			i++;
			if (i < argc)
			{
				args.out_height = stoi(argv[i]);
			}
			else
			{
				cerr << "height missing in resolution option." << endl;
				exit(EXIT_FAILURE);
			}
		}
		else
		{
			if (*(argv[i]) == '-')
			{
				cerr << "unknown argument \"" << argv[i] << "\"\n" << USAGE_TEXT << endl;
				exit(EXIT_FAILURE);
			}
			else
			{
				inc++;
				if (inc == 1)
					args.in_path_1 = argv[i];
				else if (inc == 2)
					args.in_path_2 = argv[i];
				else
				{
					cerr << "unknown argument or too many input files.\n" << USAGE_TEXT << endl;
					exit(EXIT_FAILURE);
				}
			}
		}
	}

	// check if map is provided
	if (args.map_path.empty() && args.param_path.empty() && args.parameter.radius_front == 0)
	{
		cerr << "mappingfile has to be set or parameters have to be privided by file or commandline." << endl;
		exit(EXIT_FAILURE);
	}

	// check if output resolution is requested with premade mapping table
	if (args.out_width != 0 && !args.map_path.empty())
	{
		cerr << "Changing the output resolution is only supported if calibration parameters are provided." << endl;
		exit(EXIT_FAILURE);
	}

	// set default output if not provided
	if (args.out_path.empty())
	{
		args.out_path = "equirectangular";
		if (args.prog_mode == program_mode::VIDEO)
			args.out_extension = ".mkv";
		else
			args.out_extension = ".png";
	}

	// always show preview in interactive mode
	if (args.prog_mode == program_mode::INTERACTIVE)
		args.preview = true;

	// default search range for interactive mode
	if (args.prog_mode == program_mode::INTERACTIVE && args.search_range == 0)
		args.search_range = 10;

	// print input configuration
	if (inc == 1)
		cout << "Processing video file \"" << args.in_path_1;
	else
		cout << "Processing video files \"" << args.in_path_1 << "\" and \"" << args.in_path_2;

	if (!args.map_path.empty())
		cout << "\" with mapping table \"" << args.map_path;
	else if (!args.param_path.empty())
		cout << "\" with parameter file \"" << args.param_path;

	cout << "\" as \"" << args.out_path << args.out_extension << "\"." << endl;

	DBG(
		cout << "parsed arguments:"
			 << "\n  mode:               " << args.prog_mode
			 << "\n  input:              "
			 << "\n    1:                " << args.in_path_1
			 << "\n    2:                " << args.in_path_2
			 << "\n  parameters:         "
			 << "\n    map:              " << args.map_path
			 << "\n    path:             " << args.param_path
			 << "\n    front:            "
			 << "\n      offset:         "
			 << "\n        x:            " << args.parameter.x_off_front
			 << "\n        y:            " << args.parameter.y_off_front
			 << "\n      rotation:       " << args.parameter.rot_front
			 << "\n      radius:         " << args.parameter.radius_front
			 << "\n    rear:             "
			 << "\n      offset:         "
			 << "\n        x:            " << args.parameter.x_off_rear
			 << "\n        y:            " << args.parameter.y_off_rear
			 << "\n      rotation:       " << args.parameter.rot_rear
			 << "\n      radius:         " << args.parameter.radius_rear
			 << "\n    front limit:      " << args.parameter.front_limit
			 << "\n    blend limit:      " << args.parameter.blend_limit
			 << "\n    blend size:       " << args.parameter.blend_size
			 << "\n    orientation:      " << args.parameter.orientation
			 << "\n  output:             "
			 << "\n    path:             " << args.out_path
			 << "\n    extension:        " << args.out_extension
			 << "\n    codec:            " << args.out_codec
			 << "\n    width:            " << args.out_width
			 << "\n    height:           " << args.out_height
			 << "\n  interpolation type: " << args.interpol_t
			 << "\n  search range:       " << args.search_range
			 << "\n  frameskip:          " << args.frameskip
			 << "\n  number frames:      " << args.num_frames
			 << "\n  preview:            " << args.preview
			 << "\n  video indexing:     " << args.video_index
			 << "\n  search region:      "
			 << "\n    x:                " << args.search_x
			 << "\n    y:                " << args.search_y
			 << "\n    width:            " << args.search_width
			 << "\n    height:           " << args.search_height
			 << endl;
	)
}

/**
 * @brief Remaps inputs to output based on mapping table
 *
 * @param in_1 input one
 * @param in_2 input two
 * @param map mapping table
 * @param out output
 */
template <cv::InterpolationFlags interpol_t>
void remap(const cv::Mat &in_1, const cv::Mat &in_2, const cv::Mat &map, cv::Mat &out, const extra_data &extra_data)
{
	#ifdef WITH_CUDA
	if constexpr (interpol_t == cv::INTER_NEAREST)
		cuda_remap_nn(in_1.data, in_2.data, out.data);
	else if constexpr (interpol_t == cv::INTER_LINEAR)
		cuda_remap_li(in_1.data, in_2.data, out.data);
	#else
	#ifdef WITH_OPENCL
	CL_RETERR_CHECK( extra_data.cmdq.enqueueWriteBuffer(extra_data.in_1, CL_TRUE, 0, in_1.dataend - in_1.datastart, in_1.data) );
	CL_RETERR_CHECK( extra_data.cmdq.enqueueWriteBuffer(extra_data.in_2, CL_TRUE, 0, in_2.dataend - in_2.datastart, in_2.data) );
	CL_RETERR_CHECK( extra_data.cmdq.enqueueNDRangeKernel(extra_data.k, 0, extra_data.global_size, extra_data.local_size) );
	CL_RETERR_CHECK( extra_data.cmdq.finish() );
	CL_RETERR_CHECK( extra_data.cmdq.enqueueReadBuffer(extra_data.out, CL_TRUE, 0, out.dataend - out.datastart, out.data) );
	#else
	int step = in_1.cols;
#pragma omp parallel for schedule(dynamic, 2048)
	for (int i = 0; i < map.cols * map.rows; i++)
	{
		if constexpr (interpol_t == cv::INTER_NEAREST)
			out.at<cv::Vec3b>(i) = in_1.at<cv::Vec3b>(map.at<cv::Vec4d>(i)[0]) * map.at<cv::Vec4d>(i)[2] + in_2.at<cv::Vec3b>(map.at<cv::Vec4d>(i)[1]) * map.at<cv::Vec4d>(i)[3];
		else if constexpr (interpol_t == cv::INTER_LINEAR)
		{
			// cout << i << endl;
			out.at<cv::Vec3b>(i) = in_1.at<cv::Vec3b>(map.at<cv::Vec<double, 10>>(i)[0]) * map.at<cv::Vec<double, 10>>(i)[2] + in_1.at<cv::Vec3b>(map.at<cv::Vec<double, 10>>(i)[0] + 1) * map.at<cv::Vec<double, 10>>(i)[3]
								 + in_1.at<cv::Vec3b>(map.at<cv::Vec<double, 10>>(i)[0] + step) * map.at<cv::Vec<double, 10>>(i)[4] + in_1.at<cv::Vec3b>(map.at<cv::Vec<double, 10>>(i)[0] + step + 1) * map.at<cv::Vec<double, 10>>(i)[5]
								 + in_2.at<cv::Vec3b>(map.at<cv::Vec<double, 10>>(i)[1]) * map.at<cv::Vec<double, 10>>(i)[6] + in_2.at<cv::Vec3b>(map.at<cv::Vec<double, 10>>(i)[1] + 1) * map.at<cv::Vec<double, 10>>(i)[7]
								 + in_2.at<cv::Vec3b>(map.at<cv::Vec<double, 10>>(i)[1] + step) * map.at<cv::Vec<double, 10>>(i)[8] + in_2.at<cv::Vec3b>(map.at<cv::Vec<double, 10>>(i)[1] + step + 1) * map.at<cv::Vec<double, 10>>(i)[9];
		}
	}
	#endif
	#endif
}

void remap(const cv::Mat &in_1, const cv::Mat &in_2, const cv::Mat &map, cv::InterpolationFlags interpol_t, cv::Mat &out, extra_data &extra_data)
{
	switch (interpol_t)
	{
	case cv::INTER_NEAREST:
		remap<cv::INTER_NEAREST>(in_1, in_2, map, out, extra_data);
		break;
	case cv::INTER_LINEAR:
		remap<cv::INTER_LINEAR>(in_1, in_2, map, out, extra_data);
		break;
	}
}

/**
 * @brief Remaps input to output based on mapping table
 *
 * @tparam interpol_t interpolation type to use
 * @param in input image
 * @param map mapping table
 * @param out output image
 */
template <cv::InterpolationFlags interpol_t>
void remap(const cv::Mat &in, const cv::Mat &map, cv::Mat &out, const extra_data &extra_data)
{
	#ifdef WITH_CUDA
	if constexpr (interpol_t == cv::INTER_NEAREST)
		cuda_remap_nn(in.data, out.data);
	else if constexpr (interpol_t == cv::INTER_LINEAR)
		cuda_remap_li(in.data, out.data);
	#else
	#ifdef WITH_OPENCL
	CL_RETERR_CHECK( extra_data.cmdq.enqueueWriteBuffer(extra_data.in_1, CL_TRUE, 0, in.dataend - in.datastart, in.data) );
	CL_RETERR_CHECK( extra_data.cmdq.enqueueNDRangeKernel(extra_data.k, 0, extra_data.global_size, extra_data.local_size) );
	CL_RETERR_CHECK( extra_data.cmdq.finish() );
	CL_RETERR_CHECK( extra_data.cmdq.enqueueReadBuffer(extra_data.out, CL_TRUE, 0, out.dataend - out.datastart, out.data) );
	#else
	remap<interpol_t>(in, in, map, out, extra_data);
	#endif
	#endif
}

void remap(const cv::Mat &in, const cv::Mat &map, cv::InterpolationFlags interpol_t, cv::Mat &out, extra_data &extra_data)
{
	switch (interpol_t)
	{
	case cv::INTER_NEAREST:
		remap<cv::INTER_NEAREST>(in, map, out, extra_data);
		break;
	case cv::INTER_LINEAR:
		remap<cv::INTER_LINEAR>(in, map, out, extra_data);
		break;
	}
}

template <cv::InterpolationFlags interpol_t, bool single_input>
cv::Mat get_sharpest_image(cv::VideoCapture &in_1, cv::VideoCapture &in_2, const cv::Mat &map, extra_data &extra_data, unsigned short before = 10, unsigned short after = 10, cv::Rect region = cv::Rect(0, 0, 0, 0))
{
	cv::Mat frame_1, frame_2, grey, laplace, mean, stddev, eqr_frame, best_frame;
	double best_stddev = 0;

	eqr_frame.create(map.rows, map.cols, CV_8UC3);

	in_1.set(cv::CAP_PROP_POS_FRAMES, in_1.get(cv::CAP_PROP_POS_FRAMES) - before - 1);
	if constexpr (!single_input)
		in_2.set(cv::CAP_PROP_POS_FRAMES, in_2.get(cv::CAP_PROP_POS_FRAMES) - before - 1);

	for (int i = 0; i < before + after; i++)
	{
		in_1 >> frame_1;
		if constexpr (single_input)
		{
			remap<interpol_t>(frame_1, map, eqr_frame, extra_data);
		}
		else
		{
			in_2 >> frame_2;
			remap<interpol_t>(frame_1, frame_2, map, eqr_frame, extra_data);
		}
		cv::cvtColor((region.width != 0 && region.height != 0) ? eqr_frame(region) : eqr_frame, grey, cv::COLOR_BGR2GRAY);
		cv::Laplacian(grey, laplace, CV_64F);
		cv::meanStdDev(laplace, mean, stddev);
		DBG(cout << i << ", " << stddev << endl;)
		if (stddev.at<double>(0) > best_stddev)
		{
			best_frame = eqr_frame;
			best_stddev = stddev.at<double>(0);
		}
	}

	in_1.set(cv::CAP_PROP_POS_FRAMES, in_1.get(cv::CAP_PROP_POS_FRAMES) - after + 1);
	if constexpr (!single_input)
		in_2.set(cv::CAP_PROP_POS_FRAMES, in_2.get(cv::CAP_PROP_POS_FRAMES) - after + 1);

	DBG(cout << "best stddev: " << best_stddev << endl;)
	return best_frame;
}

template <cv::InterpolationFlags interpol_t, bool single_input, program_mode prog_m>
void process_video(cv::VideoCapture &in_1, cv::VideoCapture &in_2, const cv::Mat &map, cv::Mat &equiframe, cv::VideoWriter &wrt, const program_args &args, extra_data &extra_data)
{
	const int retries = 300;
	cv::Mat frame_1, frame_2, grey, laplace, mean, stddev, best_frame;
	cv::Size outres(args.out_width, args.out_height);
	cv::Rect search_region(args.search_x * args.out_width, args.search_y * args.out_height,
						   args.search_width * args.out_width, args.search_height * args.out_height);
	double best_stddev = 0;
	int key, out_idx = 0, best_idx;
	size_t search_count = args.search_range / 2;
	char full_out_path[args.out_path.size() + args.out_extension.size() + args.out_dec_len];
	bool playing = prog_m != program_mode::INTERACTIVE, play_once = true;
	TIMES
	(
		size_t mapping_count = 0, loading_count = 0, preview_count = 0, malloc_count = 0, write_count = 0, skip_count = 0, laplace_count = 0, loop_count = 0;
		double mapping_sum = 0, loading_sum = 0, preview_sum = 0, malloc_sum = 0, write_sum = 0, skip_sum = 0, laplace_sum = 0, loop_sum = 0;
		std::chrono::steady_clock::time_point mapping_start, mapping_end, loading_start, loading_end, preview_start, preview_end,
											  malloc_start, malloc_end, write_start, write_end, skip_start, skip_end, laplace_start, laplace_end,
											  loop_start, loop_end;
	)
	// #pragma omp parallel
	// {
	// #pragma omp single
	while ((key = cv::waitKey(1)) != 27)
	{
		TIMES(loop_start = std::chrono::steady_clock::now();)
		if (playing || play_once)
		{
			TIMES(loading_start = std::chrono::steady_clock::now();)
			// get a new frame from camera
			if constexpr (single_input)
			{
				in_1 >> frame_1;
				if (frame_1.empty())
					break;
			}
			else
			{
				in_1 >> frame_1;
				in_2 >> frame_2;
				for (int i = 0; i < retries && frame_1.empty(); i++)
					in_1 >> frame_1;
				for (int i = 0; i < retries && frame_2.empty(); i++)
					in_2 >> frame_2;
				if (frame_1.empty() || frame_2.empty())
					break;
			}
			TIMES(loading_end = std::chrono::steady_clock::now(); loading_count++;
				  loading_sum += std::chrono::duration_cast<std::chrono::duration<double>>(loading_end - loading_start).count();)

			TIMES(malloc_start = std::chrono::steady_clock::now();)
			equiframe = cv::Mat(args.out_height, args.out_width, CV_8UC3);
			TIMES(malloc_end = std::chrono::steady_clock::now(); malloc_count++;
				  malloc_sum += std::chrono::duration_cast<std::chrono::duration<double>>(malloc_end - malloc_start).count();)

			TIMES(mapping_start = std::chrono::steady_clock::now();)
			if constexpr (single_input)
				remap<interpol_t>(frame_1, map, equiframe, extra_data);
			else
				remap<interpol_t>(frame_1, frame_2, map, equiframe, extra_data);
			TIMES(mapping_end = std::chrono::steady_clock::now(); mapping_count++;
				  mapping_sum += std::chrono::duration_cast<std::chrono::duration<double>>(mapping_end - mapping_start).count();)

			TIMES(preview_start = std::chrono::steady_clock::now();)
			// Auf den Schirm Spoki
			if (args.preview)
				imshow(WINDOW_NAME, equiframe);
			TIMES(preview_end = std::chrono::steady_clock::now(); preview_count++;
				  preview_sum += std::chrono::duration_cast<std::chrono::duration<double>>(preview_end - preview_start).count();)

			play_once = false;
		}
		if constexpr (prog_m == program_mode::INTERACTIVE)
		{
			switch (key)
			{
			case ' ':
				playing ^= 0x01;
				break;
			case 'k':
				playing ^= 0x01;
				break;
			case 'f':
				playing = false;
				if (args.search_range > 0)
					equiframe = get_sharpest_image<interpol_t, single_input>(in_1, in_2, map, extra_data, args.search_range / 2, args.search_range / 2, search_region);
				imshow(WINDOW_NAME, equiframe);
				break;
			case 's':
				sprintf(full_out_path, (args.out_path + args.out_extension).c_str(), out_idx);
				#pragma omp task firstprivate(full_out_path, equiframe)
				cv::imwrite(full_out_path, equiframe);
				out_idx++;
				break;
			case '.':
				play_once = true;
				break;
			case ',':
				in_1.set(cv::CAP_PROP_POS_FRAMES, in_1.get(cv::CAP_PROP_POS_FRAMES) - 2);
				if constexpr (!single_input)
					in_2.set(cv::CAP_PROP_POS_FRAMES, in_2.get(cv::CAP_PROP_POS_FRAMES) - 2);
				play_once = true;
				break;
			case 83: // arrow right
				in_1.set(cv::CAP_PROP_POS_MSEC, in_1.get(cv::CAP_PROP_POS_MSEC) + 5000);
				if constexpr (!single_input)
					in_2.set(cv::CAP_PROP_POS_MSEC, in_2.get(cv::CAP_PROP_POS_MSEC) + 5000);
				play_once = true;
				break;
			case 81: // arrow left
				in_1.set(cv::CAP_PROP_POS_MSEC, in_1.get(cv::CAP_PROP_POS_MSEC) - 5000);
				if constexpr (!single_input)
					in_2.set(cv::CAP_PROP_POS_MSEC, in_2.get(cv::CAP_PROP_POS_MSEC) - 5000);
				play_once = true;
				break;
			case 'l':
				in_1.set(cv::CAP_PROP_POS_MSEC, in_1.get(cv::CAP_PROP_POS_MSEC) + 10000);
				if constexpr (!single_input)
					in_2.set(cv::CAP_PROP_POS_MSEC, in_2.get(cv::CAP_PROP_POS_MSEC) + 10000);
				play_once = true;
				break;
			case 'j':
				in_1.set(cv::CAP_PROP_POS_MSEC, in_1.get(cv::CAP_PROP_POS_MSEC) - 10000);
				if constexpr (!single_input)
					in_2.set(cv::CAP_PROP_POS_MSEC, in_2.get(cv::CAP_PROP_POS_MSEC) - 10000);
				play_once = true;
				break;
			default:
				break;
			}
		}
		else
		{
			TIMES(laplace_start = std::chrono::steady_clock::now();)
			if (args.search_range > 0)
			{
				cv::cvtColor(equiframe(search_region), grey, cv::COLOR_BGR2GRAY);
				cv::Laplacian(grey, laplace, CV_64F);
				cv::meanStdDev(laplace, mean, stddev);
				DBG(cout << "stddev: " << stddev << endl;)
				if (stddev.at<double>(0) > best_stddev)
				{
					best_frame = equiframe;
					best_stddev = stddev.at<double>(0);
					best_idx = in_1.get(cv::CAP_PROP_POS_FRAMES) - 1;
				}
				search_count++;
			}
			else
			{
				best_frame = equiframe;
				best_idx = in_1.get(cv::CAP_PROP_POS_FRAMES) - 1;
			}
			TIMES(laplace_end = std::chrono::steady_clock::now(); laplace_count++;
				  laplace_sum += std::chrono::duration_cast<std::chrono::duration<double>>(laplace_end - laplace_start).count();)

			if (search_count == args.search_range)
			{
				DBG( cout << "best stddev: " << best_stddev << endl; )
				TIMES(write_start = std::chrono::steady_clock::now();)
				if constexpr (prog_m == program_mode::VIDEO)
				{
					// #pragma omp taskwait
					// #pragma omp task firstprivate(best_frame)
					wrt.write(best_frame);
				}
				if constexpr (prog_m == program_mode::IMAGE)
				{
					if (args.video_index)
						out_idx = best_idx;
					sprintf(full_out_path, (args.out_path + args.out_extension).c_str(), out_idx);
					// #pragma omp task firstprivate(full_out_path, best_frame)
					cv::imwrite(full_out_path, best_frame);
					DBG(cout << "Saved image: " << full_out_path << endl;)
					out_idx++;
				}
				TIMES(write_end = std::chrono::steady_clock::now(); write_count++;
					write_sum += std::chrono::duration_cast<std::chrono::duration<double>>(write_end - write_start).count();)
				search_count = 0;
				best_stddev = 0;
				TIMES(skip_start = std::chrono::steady_clock::now();)
				if constexpr (single_input)
				{
					for (size_t i = 0; i < args.frameskip - args.search_range; i++)
					{
						in_1.grab();
					}
				}
				else
				{
					for (size_t i = 0; i < args.frameskip - args.search_range; i++)
					{
						in_1.grab();
						in_2.grab();
					}
				}
				TIMES(skip_end = std::chrono::steady_clock::now(); skip_count++;
					skip_sum += std::chrono::duration_cast<std::chrono::duration<double>>(skip_end - skip_start).count();)
			}
		}
		TIMES(loop_end = std::chrono::steady_clock::now(); loop_count++;
				loop_sum += std::chrono::duration_cast<std::chrono::duration<double>>(loop_end - loop_start).count();)

		// print times
		TIMES
		(
			if ((loading_count % 100) == 0 && loading_count != 0) {
				printf("frame nr: %lu\n", loading_count);
				printf("%8.4fms avg loading time\n", loading_sum / loading_count * 1000);
				printf("%8.4fms avg malloc time\n", malloc_sum / malloc_count * 1000);
				printf("%8.4fms avg mapping time\n", mapping_sum / mapping_count * 1000);
				printf("%8.4fms avg preview time\n", preview_sum / preview_count * 1000);
				if constexpr (prog_m != program_mode::INTERACTIVE)
				{
					printf("%8.4fms avg laplace time\n", laplace_sum / laplace_count * 1000);
					printf("%8.4fms avg write time\n", write_sum / write_count * 1000);
					printf("%8.4fms avg skip time\n", skip_sum / skip_count * 1000);
				}
				printf("%8.4fms avg loop time\n", loop_sum / 100. * 1000);
				printf("%8.4fhz avg full loop\n", 100. / loop_sum);
				printf("\n");
				loop_sum = 0;
			}
		)
	}
	// }
}

template <cv::InterpolationFlags interpol_t>
void process_input(program_args &args)
{
	cv::Mat frame_1, frame_2, equiframe, mapping_table;
	cv::VideoCapture cap_1, cap_2;
	cv::VideoWriter wrt;

	extra_data extra_data;

	// create windows
	if (args.preview)
	{
		cv::namedWindow(WINDOW_NAME, cv::WINDOW_KEEPRATIO | cv::WINDOW_NORMAL);
		cv::resizeWindow(WINDOW_NAME, 1440, 720);
	}

	if (args.in_path_2.empty()) // one input
	{
		// open video
		cap_1.open(args.in_path_1);
		if (!cap_1.isOpened())
		{
			cout << "Could not open video file \"" << args.in_path_1 << "\"." << endl;
			exit(EXIT_FAILURE);
		}

		// get first frame and check if file is empty
		cap_1 >> frame_1;
		if (frame_1.empty())
		{
			cout << "Video file \"" << args.in_path_1 << "\" is empty." << endl;
			exit(EXIT_FAILURE);
		}

		// set output resolution if none is set
		if (args.out_width == 0 || args.out_height == 0)
		{
			int size = frame_1.cols / 2 < frame_1.rows ? frame_1.cols / 2 : frame_1.rows;
			args.out_width = size * 2;
			args.out_height = size;
		}

		// read / generate mapping table
		if (!args.map_path.empty())
		{
			read_mapping_file<interpol_t, true>(args.map_path, mapping_table);
			// check resolution
			if (frame_1.cols != mapping_table.cols || frame_1.rows != mapping_table.rows)
			{
				cout << "Image(" << frame_1.cols << 'x' << frame_1.rows
					<< ") and map(" << mapping_table.cols << 'x' << mapping_table.rows << ") resolution doesn't match." << endl;
				exit(EXIT_FAILURE);
			}
		}
		else
		{
			if (!args.param_path.empty())
				read_mapping_params(args.param_path, args.parameter);
			gen_equi_mapping_table<interpol_t, true, true>(args.out_width, args.out_height, frame_1.cols / 2, frame_1.rows, &args.parameter, mapping_table);
		}

		#ifdef WITH_CUDA
		// allocate device memory and copy mapping table
		init_device_memory(mapping_table.data, mapping_table.cols, mapping_table.rows);
		#endif
		#ifdef WITH_OPENCL
		cl_init_device(interpol_t, true, mapping_table, frame_1.cols / 2, frame_1.rows, string((char*) src_mapping_cl, src_mapping_cl_len), &extra_data);
		#endif

		// init output mat
		equiframe.create(mapping_table.rows, mapping_table.cols, CV_8UC3);

		// process single image
		cap_1 >> frame_2;
		if (frame_2.empty())
		{
			remap<interpol_t>(frame_1, mapping_table, equiframe, extra_data);
			cv::imshow(WINDOW_NAME, equiframe);
			if (!args.out_path.empty())
				cv::imwrite(args.out_path+args.out_extension, equiframe);
			cv::waitKey();
			return;
		}

		// calculate (max) number of output frames
		if (args.num_frames > 0)
			args.frameskip = cap_1.get(cv::CAP_PROP_FRAME_COUNT) / args.num_frames - 1;
		else if (args.frameskip > 0)
			args.num_frames = cap_1.get(cv::CAP_PROP_FRAME_COUNT) / (args.frameskip + 1) + 1;
		else
			args.num_frames = cap_1.get(cv::CAP_PROP_FRAME_COUNT);

		// check if search ranges collide
		if (args.prog_mode != program_mode::INTERACTIVE && args.frameskip < args.search_range)
		{
			cout << "search range(" << args.search_range << ") has to be less then frameskip(" << args.frameskip << ")." << endl;
			exit(EXIT_FAILURE);
		}

		// calculate max number of decimal digits needed for frame index
		if ((args.prog_mode == program_mode::INTERACTIVE || args.prog_mode == program_mode::IMAGE)
			&& args.out_dec_len == 0)
		{
			string s_num_idx;
			if (args.video_index)
				s_num_idx = to_string((size_t) cap_1.get(cv::CAP_PROP_FRAME_COUNT));
			else
				s_num_idx = to_string(args.num_frames);
			args.out_dec_len = s_num_idx.size();
			args.out_path += "%0" + to_string(args.out_dec_len) + 'd';
		}

		// jump back to beginning
		cap_1.set(cv::CAP_PROP_POS_FRAMES, 0);

		// do mapping
		switch (args.prog_mode)
		{
			case program_mode::INTERACTIVE:
				process_video<interpol_t, true, program_mode::INTERACTIVE>(cap_1, cap_2, mapping_table, equiframe, wrt, args, extra_data);
				break;
			case program_mode::VIDEO:
				wrt.open(args.out_path+args.out_extension, args.out_codec == 0 ? cap_1.get(cv::CAP_PROP_FOURCC) : args.out_codec, cap_1.get(cv::CAP_PROP_FPS),
						cv::Size(args.out_width, args.out_height));
				if (!wrt.isOpened())
				{
					cerr << "Couldn't open \"" << args.out_path+args.out_extension << "\" for writing." << endl;
					exit(EXIT_FAILURE);
				}
				process_video<interpol_t, true, program_mode::VIDEO>(cap_1, cap_2, mapping_table, equiframe, wrt, args, extra_data);
				break;
			case program_mode::IMAGE:
				process_video<interpol_t, true, program_mode::IMAGE>(cap_1, cap_2, mapping_table, equiframe, wrt, args, extra_data);
				break;
			default:
				cerr << "Program Mode not supported." << endl;
				exit(EXIT_FAILURE);
				break;
		}
	}
	else // two inputs
	{
		// open video
		cap_1.open(args.in_path_1);
		cap_2.open(args.in_path_2);
		if (!cap_1.isOpened())
			cout << "Could not open video file \"" << args.in_path_1 << "\"." << endl;
		if (!cap_2.isOpened())
			cout << "Could not open video file \"" << args.in_path_2 << "\"." << endl;
		if (!cap_1.isOpened() || !cap_2.isOpened())
			exit(EXIT_FAILURE);

		// get first frame and check if file is empty
		cap_1 >> frame_1;
		cap_2 >> frame_2;
		if (frame_1.empty())
			cout << "Video file \"" << args.in_path_1 << "\" is empty." << endl;
		if (frame_2.empty())
			cout << "Video file \"" << args.in_path_2 << "\" is empty." << endl;
		if (frame_1.empty() || frame_2.empty())
			exit(EXIT_FAILURE);

		// set output resolution if none is set
		if (args.out_width == 0 || args.out_height == 0)
		{
			int size = frame_1.cols < frame_1.rows ? frame_1.cols : frame_1.rows;
			args.out_width = size * 2;
			args.out_height = size;
		}

		// read / generate mapping table
		if (!args.map_path.empty())
		{
			read_mapping_file<interpol_t, false>(args.map_path, mapping_table);
			// check resolution
			if (frame_1.cols * 2 != mapping_table.cols || frame_1.rows != mapping_table.rows || frame_2.cols * 2 != mapping_table.cols || frame_2.rows != mapping_table.rows)
			{
				cout << "Image1(" << frame_1.cols << 'x' << frame_1.rows << "), Image2(" << frame_2.cols << 'x' << frame_2.rows
					<< ") and map(" << mapping_table.cols << 'x' << mapping_table.rows << ") resolution doesn't match." << endl;
				exit(EXIT_FAILURE);
			}
		}
		else
		{
			if (!args.param_path.empty())
				read_mapping_params(args.param_path, args.parameter);
			gen_equi_mapping_table<interpol_t, false, true>(args.out_width, args.out_height, frame_1.cols, frame_1.rows, &args.parameter, mapping_table);
		}

		#ifdef WITH_CUDA
		// allocate device memory and copy mapping table
		init_device_memory(mapping_table.data, mapping_table.cols, mapping_table.rows);
		#endif
		#ifdef WITH_OPENCL
		cl_init_device(interpol_t, false, mapping_table, frame_1.cols, frame_1.rows, string((char*) src_mapping_cl, src_mapping_cl_len), &extra_data);
		#endif

		// init output mat
		equiframe.create(mapping_table.rows, mapping_table.cols, CV_8UC3);

		// process single image
		if (!cap_1.grab())
		{
			remap<interpol_t>(frame_1, frame_2, mapping_table, equiframe, extra_data);
			cv::imshow(WINDOW_NAME, equiframe);
			if (!args.out_path.empty())
				cv::imwrite(args.out_path+args.out_extension, equiframe);
			cv::waitKey();
			return;
		}

		// // check length
		// if (cap_1.get(cv::CAP_PROP_FRAME_COUNT) != cap_2.get(cv::CAP_PROP_FRAME_COUNT))
		// {
		// 	cerr << "Input 1 and 2 differ in length." << endl;
		// 	exit(EXIT_FAILURE);
		// }

		// calculate (max) number of output frames
		if (args.num_frames > 0)
			args.frameskip = cap_1.get(cv::CAP_PROP_FRAME_COUNT) / args.num_frames - 1;
		else if (args.frameskip > 0)
			args.num_frames = cap_1.get(cv::CAP_PROP_FRAME_COUNT) / (args.frameskip + 1) + 1;
		else
			args.num_frames = cap_1.get(cv::CAP_PROP_FRAME_COUNT);

		// check if search ranges collide
		if (args.prog_mode != program_mode::INTERACTIVE && args.frameskip < args.search_range)
		{
			cout << "search range(" << args.search_range << ") has to be less then frameskip(" << args.frameskip << ")." << endl;
			exit(EXIT_FAILURE);
		}

		// calculate max number of decimal digits needed for frame index
		if ((args.prog_mode == program_mode::INTERACTIVE || args.prog_mode == program_mode::IMAGE)
			&& args.out_dec_len == 0)
		{
			string s_num_idx;
			if (args.video_index)
				s_num_idx = to_string((size_t) cap_1.get(cv::CAP_PROP_FRAME_COUNT));
			else
				s_num_idx = to_string(args.num_frames);
			args.out_dec_len = s_num_idx.size();
			args.out_path += "%0" + to_string(args.out_dec_len) + 'd';
		}

		// jump back to beginning
		cap_1.set(cv::CAP_PROP_POS_FRAMES, args.parameter.time_offset < 0 ? -args.parameter.time_offset : 0);
		cap_2.set(cv::CAP_PROP_POS_FRAMES, args.parameter.time_offset > 0 ? args.parameter.time_offset : 0);

		// do mapping
		switch (args.prog_mode)
		{
			case program_mode::INTERACTIVE:
				process_video<interpol_t, false, program_mode::INTERACTIVE>(cap_1, cap_2, mapping_table, equiframe, wrt, args, extra_data);
				break;
			case program_mode::VIDEO:
				wrt.open(args.out_path+args.out_extension, args.out_codec == 0 ? cap_1.get(cv::CAP_PROP_FOURCC) : args.out_codec, cap_1.get(cv::CAP_PROP_FPS),
						cv::Size(args.out_width, args.out_height));
				if (!wrt.isOpened())
				{
					cerr << "Couldn't open \"" << args.out_path+args.out_extension << "\" for writing." << endl;
					exit(EXIT_FAILURE);
				}
				process_video<interpol_t, false, program_mode::VIDEO>(cap_1, cap_2, mapping_table, equiframe, wrt, args, extra_data);
				break;
			case program_mode::IMAGE:
				process_video<interpol_t, false, program_mode::IMAGE>(cap_1, cap_2, mapping_table, equiframe, wrt, args, extra_data);
				break;
			default:
				cerr << "Program Mode not supported." << endl;
				exit(EXIT_FAILURE);
				break;
		}
	}

	// When everything done, release the video capture objects
	cap_1.release();
	cap_2.release();
	// and the video writer object
	wrt.release();
	// Closes all the frames
	cv::destroyAllWindows();
}

int main(int argc, char **argv)
{
	program_args args;
	parse_args(argc, argv, args);

	if (args.prog_mode == program_mode::INTERACTIVE)
	{
		cout << "controls:\n  space, k\tplay/pause\n  .\t\tnext frame\n  ,\t\tlast frame\n"
				"  ->\t\tjump 5s ahead\n  <-\t\tjump 5s back\n  l\t\tjump 10s ahead\n  j\t\tjump 10s back\n"
				"  f\t\tget sharpest image from surrounding frames\n  esc\t\texit" << endl;
	}
	else if (args.preview)
	{
		cout << "controls:\n  esc\t\texit" << endl;
	}

	switch (args.interpol_t)
	{
	case cv::INTER_NEAREST:
		process_input<cv::INTER_NEAREST>(args);
		break;
	case cv::INTER_LINEAR:
		process_input<cv::INTER_LINEAR>(args);
		break;
	default:
		cerr << "Interpolation type not supported." << endl;
		break;
	}

	return EXIT_SUCCESS;
}

template <bool single_input>
void mapper::init(cv::InterpolationFlags interpol_t, size_t buffer_length)
{
	this->buffer_length = buffer_length + 1;
	this->read_pos = this->buffer_length;
	this->write_pos = 0;

	this->unread = new atomic_bool[this->buffer_length + 1];
	this->unwritten = new atomic_bool[this->buffer_length];
	this->out = new cv::Mat[this->buffer_length];

	for (size_t i = 0; i < this->buffer_length; i++)
	{
		this->out[i].create(this->map.rows, this->map.cols, CV_8UC3);
		this->unwritten[i] = true;
		this->unread[i] = false;
	}

	this->alive = true;
	this->running = true;
	this->good = true;

	#ifdef WITH_OPENCL
	CL_ARGERR_CHECK( this->edata.ctxt = cl::Context(CL_DEVICE_TYPE_GPU, NULL, NULL, NULL, &_cl_error) );
	CL_ARGERR_CHECK( this->edata.cmdq = cl::CommandQueue(this->edata.ctxt, 0, &_cl_error) );
	CL_ARGERR_CHECK( this->edata.prog = cl::Program(this->edata.ctxt, string((char*) src_mapping_cl, src_mapping_cl_len), true, &_cl_error) );

	size_t elelen = this->map.rows * this->map.cols;
	this->edata.global_size = cl::NDRange(elelen % extra_data::LOCAL_SIZE == 0 ? elelen : elelen + extra_data::LOCAL_SIZE - elelen % extra_data::LOCAL_SIZE);
	CL_ARGERR_CHECK( this->edata.map = cl::Buffer(this->edata.ctxt, CL_MEM_READ_WRITE, this->map.dataend - this->map.datastart, NULL, &_cl_error) );
	if constexpr (single_input)
	{
		CL_ARGERR_CHECK( this->edata.in_1 = cl::Buffer(this->edata.ctxt, CL_MEM_READ_WRITE, elelen * 3, NULL, &_cl_error) );
	}
	else
	{
		CL_ARGERR_CHECK( this->edata.in_1 = cl::Buffer(this->edata.ctxt, CL_MEM_READ_WRITE, elelen * 3 / 2, NULL, &_cl_error) );
		CL_ARGERR_CHECK( this->edata.in_2 = cl::Buffer(this->edata.ctxt, CL_MEM_READ_WRITE, elelen * 3 / 2, NULL, &_cl_error) );
	}
	CL_ARGERR_CHECK( this->edata.out = cl::Buffer(this->edata.ctxt, CL_MEM_READ_WRITE, elelen * 3, NULL, &_cl_error) );
	switch (interpol_t)
	{
	case cv::INTER_NEAREST:
		CL_ARGERR_CHECK( this->edata.k = cl::Kernel(this->edata.prog, "remap_nearest", &_cl_error) );
		break;
	case cv::INTER_LINEAR:
		CL_ARGERR_CHECK( this->edata.k = cl::Kernel(this->edata.prog, "remap_linear", &_cl_error) );
		break;
	default:
		throw invalid_argument("Interpolation type not supported.");
		break;
	}
	CL_RETERR_CHECK( this->edata.k.setArg(0, this->edata.in_1) );
	if constexpr (single_input)
	{
		CL_RETERR_CHECK( this->edata.k.setArg(1, this->edata.in_1) );
	}
	else
	{
		CL_RETERR_CHECK( this->edata.k.setArg(1, this->edata.in_2) );
	}
	CL_RETERR_CHECK( this->edata.k.setArg(2, this->edata.out) );
	CL_RETERR_CHECK( this->edata.k.setArg(3, this->edata.map) );
	CL_RETERR_CHECK( this->edata.k.setArg(4, (unsigned int) elelen) );
	CL_RETERR_CHECK( this->edata.cmdq.enqueueWriteBuffer(this->edata.map, CL_TRUE, 0, this->map.dataend - this->map.datastart, this->map.data) );
	#endif

	switch (interpol_t)
	{
	case cv::INTER_NEAREST:
		this->worker = std::thread(&mapper::process<cv::INTER_NEAREST, single_input>, this);
		break;
	case cv::INTER_LINEAR:
		this->worker = std::thread(&mapper::process<cv::INTER_LINEAR, single_input>, this);
		break;
	default:
		throw invalid_argument("Interpolation type not supported.");
		break;
	}
}

mapper::mapper(cv::VideoCapture &in, const cv::Mat map, size_t frameskip, cv::InterpolationFlags interpol_t, size_t buffer_length)
{
	this->cap_1 = in;
	if (!this->cap_1.isOpened())
		throw invalid_argument("VideoCapture Object isn't opened.");

	this->map = map;
	if (this->map.cols != (int) this->cap_1.get(cv::CAP_PROP_FRAME_WIDTH) * 2 || this->map.rows != (int) this->cap_1.get(cv::CAP_PROP_FRAME_HEIGHT))
		throw invalid_argument("Input and mappingtable differ in size.");

	this->frameskip = frameskip;

	this->init<true>(interpol_t, buffer_length);
}

mapper::mapper(cv::VideoCapture &in_1, cv::VideoCapture in_2, const cv::Mat map, size_t frameskip, cv::InterpolationFlags interpol_t, size_t buffer_length)
{
	this->cap_1 = in_1;
	if (!this->cap_1.isOpened())
		throw invalid_argument("VideoCapture Object in_1 isn't opened.");
	if (!this->cap_2.isOpened())
		throw invalid_argument("VideoCapture Object in_2 isn't opened.");
	if (this->cap_1.get(cv::CAP_PROP_FRAME_WIDTH) != this->cap_2.get(cv::CAP_PROP_FRAME_WIDTH)
		|| this->cap_1.get(cv::CAP_PROP_FRAME_HEIGHT) != this->cap_2.get(cv::CAP_PROP_FRAME_HEIGHT))
		throw invalid_argument("Input \"in_1\" and \"in_2\" differ in size.");

	this->map = map;
	if (this->map.cols != (int) this->cap_1.get(cv::CAP_PROP_FRAME_WIDTH) * 2 || this->map.rows != (int) this->cap_1.get(cv::CAP_PROP_FRAME_HEIGHT))
		throw invalid_argument("Inputs and mappingtable differ in size.");

	this->frameskip = frameskip;

	this->init<false>(interpol_t, buffer_length);
}

mapper::mapper(const std::string &in_path, const std::string &map_path, size_t frameskip, cv::InterpolationFlags interpol_t, size_t buffer_length)
{
	if (!this->cap_1.open(in_path))
		throw invalid_argument('"' + in_path + "\" is can't be opened for reading.");

	switch (interpol_t)
	{
	case cv::INTER_NEAREST:
		read_mapping_file<cv::INTER_NEAREST, true>(map_path, this->map);
		break;
	case cv::INTER_LINEAR:
		read_mapping_file<cv::INTER_LINEAR, true>(map_path, this->map);
		break;
	default:
		throw invalid_argument("Interpolation type not supported.");
		break;
	}

	if (this->map.cols != (int) this->cap_1.get(cv::CAP_PROP_FRAME_WIDTH) || this->map.rows != (int) this->cap_1.get(cv::CAP_PROP_FRAME_HEIGHT))
		throw invalid_argument("Input \"" + in_path + "\" and mappingtable: \"" + map_path + "\" differ in size.");

	this->frameskip = frameskip;

	this->init<true>(interpol_t, buffer_length);
}

mapper::mapper(const std::string &in_path_1, const std::string &in_path_2, const std::string &map_path, size_t frameskip, cv::InterpolationFlags interpol_t, size_t buffer_length)
{
	if (!this->cap_1.open(in_path_1))
		throw invalid_argument('"' + in_path_1 + "\" is can't be opened for reading.");
	if (!this->cap_2.open(in_path_2))
		throw invalid_argument('"' + in_path_2 + "\" is can't be opened for reading.");
	if (this->cap_1.get(cv::CAP_PROP_FRAME_WIDTH) != this->cap_2.get(cv::CAP_PROP_FRAME_WIDTH)
		|| this->cap_1.get(cv::CAP_PROP_FRAME_HEIGHT) != this->cap_2.get(cv::CAP_PROP_FRAME_HEIGHT))
		throw invalid_argument("Input \"" + in_path_1 + "\" and \"" + in_path_2 + "\" differ in size.");

	switch (interpol_t)
	{
	case cv::INTER_NEAREST:
		read_mapping_file<cv::INTER_NEAREST, false>(map_path, this->map);
		break;
	case cv::INTER_LINEAR:
		read_mapping_file<cv::INTER_LINEAR, false>(map_path, this->map);
		break;
	default:
		throw invalid_argument("Interpolation type not supported.");
		break;
	}

	if (this->map.cols != (int) this->cap_1.get(cv::CAP_PROP_FRAME_WIDTH) * 2 || this->map.rows != (int) this->cap_1.get(cv::CAP_PROP_FRAME_HEIGHT))
		throw invalid_argument("Inputs and mappingtable: \"" + map_path + "\" differ in size.");

	this->frameskip = frameskip;

	this->init<false>(interpol_t, buffer_length);
}

mapper::~mapper()
{
	this->alive = false;
	this->worker.join();
	this->cap_1.release();
	this->cap_2.release();
	delete[] this->unread;
	delete[] this->unwritten;
	delete[] this->out;
}

template <cv::InterpolationFlags interpol_t, bool single_input>
void mapper::process()
{
	TIMES( chrono::steady_clock::time_point get_s, get_e, map_s, map_e; double get_sum = 0, map_sum = 0; long fc = 0; )
	while (this->alive)
	{
		if (this->running)
		{
			this->cap_lock.lock();
			TIMES( get_s = chrono::steady_clock::now(); )
			if constexpr (single_input)
			{
				this->good = this->cap_1.read(this->in_1);
				for (unsigned char i = 0; i < this->frameskip; i++)
				{
					this->cap_1.grab();
				}
			}
			else
			{
				this->good = this->cap_1.read(this->in_1) && this->cap_2.read(this->in_2);
				for (unsigned char i = 0; i < this->frameskip; i++)
				{
					this->cap_1.grab();
					this->cap_2.grab();
				}
			}
			TIMES( get_e = chrono::steady_clock::now(); get_sum += std::chrono::duration_cast<std::chrono::duration<double>>(get_e - get_s).count(); )
			this->cap_lock.unlock();

			if (this->good)
			{
				while (this->unread[this->write_pos].exchange(true))
				{
					if (!this->alive) return;
					this_thread::sleep_for(1ms);
				}

				DBG( printf("process: r%lu locked\n", this->write_pos); )
				TIMES( map_s = chrono::steady_clock::now(); )
				if constexpr (single_input)
					remap<interpol_t>(this->in_1, this->map, this->out[this->write_pos], this->edata);
				else
					remap<interpol_t>(this->in_1, this->in_2, this->map, this->out[this->write_pos], this->edata);
				TIMES( map_e = chrono::steady_clock::now(); map_sum += std::chrono::duration_cast<std::chrono::duration<double>>(map_e - map_s).count(); )
				this->unwritten[this->write_pos] = false;
				DBG( printf("process: w%lu unlocked\n", this->write_pos); )
				if (++this->write_pos == this->buffer_length)
					this->write_pos = 0;
				TIMES( fc++; if (fc % 150 == 0) {printf("\nframe: %ld\nget=%fms\nmap=%fms\n", fc, get_sum / fc * 1000, map_sum / fc * 1000);} )
			}
		}
		else
		{
			this_thread::sleep_for(100ms);
		}
	}
}

bool mapper::get_next_img(cv::Mat &img)
{
	this->unread[this->read_pos] = false;
	DBG( printf("gecv::Vec3b: r%lu unlocked\n", this->read_pos); )
	if (++this->read_pos >= this->buffer_length)
		this->read_pos = 0;

	bool lk;
	while ((lk = this->unwritten[this->read_pos].exchange(true)) && this->good)
		this_thread::sleep_for(1ms);
	if (!lk)
	{
		DBG( printf("gecv::Vec3b: w%lu locked\n", this->read_pos); )
		img = this->out[this->read_pos];
		return true;
	}
	else
		return false;
}

bool mapper::copy_next_img(cv::Mat &img)
{
	cv::Mat temp;
	bool ret = this->get_next_img(temp);
	img = temp.clone();
	return ret;
}

bool mapper::jump_mseconds(int ms)
{
	this->cap_lock.lock();
	bool ret = this->cap_1.set(cv::CAP_PROP_POS_MSEC, this->cap_1.get(cv::CAP_PROP_POS_MSEC) + ms)
		&& this->cap_2.isOpened()
		 ? this->cap_2.set(cv::CAP_PROP_POS_MSEC, this->cap_2.get(cv::CAP_PROP_POS_MSEC) + ms)
		 : true;
	this->cap_lock.unlock();
	return ret;
}

bool mapper::jump_frames(int frames)
{
	this->cap_lock.lock();
	bool ret = this->cap_1.set(cv::CAP_PROP_POS_FRAMES, this->cap_1.get(cv::CAP_PROP_POS_FRAMES) + frames - 1)
		&& this->cap_2.isOpened()
		 ? this->cap_2.set(cv::CAP_PROP_POS_FRAMES, this->cap_2.get(cv::CAP_PROP_POS_FRAMES) + frames - 1)
		 : true;
	this->cap_lock.unlock();
	return ret;
}

void mapper::set_frameskip(size_t frames)
{
	this->cap_lock.lock();
	this->frameskip = frames;
	this->cap_lock.unlock();
}

size_t mapper::get_frameskip()
{
	this->cap_lock.lock();
	size_t frames = this->frameskip;
	this->cap_lock.unlock();
	return frames;
}

void mapper::start() { this->running = true; }
void mapper::stop() { this->running = false; }

mapping::mapping(const cv::Size &in_res, bool single_input, cv::InterpolationFlags interpol_t, const cv::Size &out_res)
{
	if (out_res.width > 0 && out_res.height > 0)
		this->out_res = out_res;
	else
	{
		int size = in_res.width < in_res.height ? in_res.width : in_res.height;
		this->out_res.width = size * 2;
		this->out_res.height = size;
	}
	this->in_res = in_res;
	this->single_input = single_input;
	this->set_interpolation(interpol_t);
}

mapping::mapping(const string &param_file, const cv::Size &in_res, bool single_input, cv::InterpolationFlags interpol_t, const cv::Size &out_res)
 : mapping(in_res, single_input, interpol_t, out_res)
{
	this->load_from_param(param_file);
}

void mapping::load_from_table(const string &file)
{
	if (this->single_input)
	{
		switch (this->interpol_t)
		{
		case cv::INTER_NEAREST:
			read_mapping_file<cv::INTER_NEAREST, true>(file, this->mapping_table);
			break;
		case cv::INTER_LINEAR:
			read_mapping_file<cv::INTER_LINEAR, true>(file, this->mapping_table);
			break;
		default:
			break;
		}
	}
	else
	{
		switch (this->interpol_t)
		{
		case cv::INTER_NEAREST:
			read_mapping_file<cv::INTER_NEAREST, false>(file, this->mapping_table);
			break;
		case cv::INTER_LINEAR:
			read_mapping_file<cv::INTER_LINEAR, false>(file, this->mapping_table);
			break;
		default:
			break;
		}
	}
}

void mapping::gen_mapping_table()
{
	if (this->single_input)
	{
		switch (this->interpol_t)
		{
		case cv::INTER_NEAREST:
			gen_equi_mapping_table<cv::INTER_NEAREST, true, true>(this->out_res.width, this->out_res.height, this->in_res.width, this->in_res.height, &(this->params), this->mapping_table);
			break;
		case cv::INTER_LINEAR:
			gen_equi_mapping_table<cv::INTER_LINEAR, true, true>(this->out_res.width, this->out_res.height, this->in_res.width, this->in_res.height, &(this->params), this->mapping_table);
			break;
		default:
			break;
		}
	}
	else
	{
		switch (this->interpol_t)
		{
		case cv::INTER_NEAREST:
			gen_equi_mapping_table<cv::INTER_NEAREST, false, true>(this->out_res.width, this->out_res.height, this->in_res.width, this->in_res.height, &(this->params), this->mapping_table);
			break;
		case cv::INTER_LINEAR:
			gen_equi_mapping_table<cv::INTER_LINEAR, false, true>(this->out_res.width, this->out_res.height, this->in_res.width, this->in_res.height, &(this->params), this->mapping_table);
			break;
		default:
			break;
		}
	}
	#ifdef WITH_OPENCL
	cl_init_device(this->interpol_t, this->single_input, this->mapping_table, this->in_res.width, this->in_res.height, std::string((char*)src_mapping_cl, src_mapping_cl_len), &(this->data));
	#endif
}

void mapping::load_from_param(const string &file)
{
	read_mapping_params(file, this->params);
	this->gen_mapping_table();
}


void mapping::build_from_param(const calib_params &params)
{
	this->params = params;
	this->gen_mapping_table();
}

void mapping::set_interpolation(cv::InterpolationFlags interpol)
{
	this->interpol_t = interpol;
	switch (this->interpol_t)
	{
	case cv::INTER_NEAREST:
		this->remap_fnc_s = remap<cv::INTER_NEAREST>;
		this->remap_fnc_d = remap<cv::INTER_NEAREST>;
		break;
	case cv::INTER_LINEAR:
		this->remap_fnc_s = remap<cv::INTER_LINEAR>;
		this->remap_fnc_d = remap<cv::INTER_LINEAR>;
		break;
	default:
		break;
	}
	// if (!this->mapping_table.data)
		this->gen_mapping_table();
}
cv::InterpolationFlags mapping::get_interpolation() const
{
	return this->interpol_t;
}

void mapping::set_output_resolution(const cv::Size &res)
{
	this->out_res = res;
	// if (!this->mapping_table.data)
		this->gen_mapping_table();
}
const cv::Size& mapping::get_output_resolution() const
{
	return this->out_res;
}

void mapping::map(const cv::Mat &in, cv::Mat &out) const
{
	this->remap_fnc_s(in, this->mapping_table, out, this->data);
}
void mapping::map(const cv::Mat &in_1, const cv::Mat &in_2, cv::Mat &out) const
{
	this->remap_fnc_d(in_1, in_2, this->mapping_table, out, this->data);
}
