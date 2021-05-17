#include <iostream>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "mapping.h"
#ifdef WITH_OPENCL
#include <CL/cl2.hpp>
#include "mapping.cl.h"
#include "clerror.h"
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
	string in_path_1, in_path_2, map_path, out_path, out_extension;
	interpolation_type interpol_t = interpolation_type::NEAREST_NEIGHBOUR;
	int out_codec = 0, out_dec_len = 0;
	size_t frameskip = 0, num_frames = 0, search_range = 0;
	program_mode prog_mode = program_mode::INTERACTIVE;
};

/**
 * @brief Reads the mapping table from file at path
 * 
 * @tparam interpol_t interpolation type to buld the mapping table for
 * @param path path to mapping file
 * @param mapping_table mapping table in format [x1, y1, x2, y2, blend factor 1, blend factor 2]
 * 						or [x1, y1, x2, y2, blend factor 1, blend factor 2, x1 y1 factor, x1+1 y1 factor, x1 y1+1 factor, x1+1 y1+1 factor, x2 y2 factor, x2+1 y2 factor, x2 y2+1 factor, x2+1 y2+1 factor]
 */
template <interpolation_type interpol_t, bool single_input>
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
				if (mte[0] < 0 || mte[0] >= height || mte[1] < 0 || mte[1] >= height || mte[2] < 0 || mte[2] >= height || mte[3] < 0 || mte[3] >= height || mte[4] < 0 || mte[4] > 1)
				{
					cerr << "Error reading mapping file \"" << path << "\" on line " << j * width + i + 2 << ", value out of range." << endl;
					exit(EXIT_FAILURE);
				}
				mte[5] = 1 - mte[4];
				if constexpr (interpol_t == interpolation_type::BILINEAR)
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
				else if constexpr (interpol_t == interpolation_type::NEAREST_NEIGHBOUR)
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
}

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
							 + argv[0] + string(" MODE [-li] [-sr <range>] [-fs <frameskip> | -nf <number frames>] [-c <codec>] [-o <output file>] -m <map file> <input file 1> [<input file 2>]\n");
	const string HELP_TEXT = "modes:\n"
							 "  interactive             int  interactively jump through the video and save images\n"
							 "  video                   vid  convert the input to a video file\n"
							 "  image                   img  convert the input to an image sequence\n"
							 "\n"
							 "global options:\n"
							 "  --help                  -h   print this message\n"
							 "  --mappingfile  --map    -m   mapping file path\n"
							 "  --output                -o   output file path with printf format for image numbers\n"
							 "  --linear-interpolation  -li  enable linear interpolation\n"
							 "  --search-range          -sr  range to search befor and after the current frame for sharper image\n"
							 "video options:\n"
							 "  --codec  --fourcc       -c   codec for video output\n"
							 "  --frameskip             -fs  number of frames to skip between each image\n"
							 "  --number-frames         -nf  number of frames to output with even spacing from start to end\n"
							 "image options:\n"
							 "  --frameskip             -fs  number of frames to skip between each image\n"
							 "  --number-frames         -nf  number of frames to output with even spacing from start to end\n";

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
			args.interpol_t = interpolation_type::BILINEAR;
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
		else if (string("-sr") == argv[i] || string("--search-range") == argv[i])
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
	if (args.map_path.empty())
	{
		cerr << "mappingfile has to be set." << endl;
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

	// default search range for interactive mode
	if (args.prog_mode == program_mode::INTERACTIVE && args.search_range == 0)
		args.search_range = 10;
	
	// print input configuration
	if (inc == 1)
		cout << "Processing video file \"" << args.in_path_1 << "\" with mapping table \"" << args.map_path << "\"." << endl;
	else
		cout << "Processing video files \"" << args.in_path_1 << "\" and \"" << args.in_path_2 << "\" with mapping table \"" << args.map_path << "\"." << endl;

	DBG(
		cout << "program arguments:"
			 << "\n  mode:               " << args.prog_mode
			 << "\n  input 1:            " << args.in_path_1
			 << "\n  input 2:            " << args.in_path_2
			 << "\n  map:                " << args.map_path
			 << "\n  output:             " << args.out_path
			 << "\n  output extension:   " << args.out_extension
			 << "\n  output codec:       " << args.out_codec
			 << "\n  interpolation type: " << args.interpol_t
			 << "\n  search range:       " << args.search_range
			 << "\n  frameskip:          " << args.frameskip
			 << "\n  number frames:      " << args.num_frames
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
template <interpolation_type interpol_t>
void remap(const cv::Mat &in_1, const cv::Mat &in_2, const cv::Mat &map, cv::Mat &out, extra_data &extra_data)
{
	#ifdef WITH_CUDA
	if constexpr (interpol_t == interpolation_type::NEAREST_NEIGHBOUR)
		cuda_remap_nn(in_1.data, in_2.data, out.data);
	else if constexpr (interpol_t == interpolation_type::BILINEAR)
		cuda_remap_li(in_1.data, in_2.data, out.data);
	#else
	#ifdef WITH_OPENCL
	CL_ERROR_CHECK( extra_data.cmdq.enqueueWriteBuffer(extra_data.in_1, CL_TRUE, 0, in_1.dataend - in_1.datastart, in_1.data) );
	CL_ERROR_CHECK( extra_data.cmdq.enqueueWriteBuffer(extra_data.in_2, CL_TRUE, 0, in_2.dataend - in_2.datastart, in_2.data) );
	CL_ERROR_CHECK( extra_data.cmdq.enqueueNDRangeKernel(extra_data.k, 0, extra_data.global_size, extra_data.local_size) );
	CL_ERROR_CHECK( extra_data.cmdq.finish() );
	CL_ERROR_CHECK( extra_data.cmdq.enqueueReadBuffer(extra_data.out, CL_TRUE, 0, out.dataend - out.datastart, out.data) );
	#else
	cv::Vec<double, 14> mte;
#pragma omp parallel for collapse(2) schedule(dynamic, 2048) private(mte)
	for (int i = 0; i < map.rows; i++)
	{
		for (int j = 0; j < map.cols; j++)
		{
			mte = map.at<cv::Vec<double, 14>>(i, j);
			if constexpr (interpol_t == interpolation_type::NEAREST_NEIGHBOUR)
				out.at<cv::Vec3b>(i, j) = in_1.at<cv::Vec3b>(mte[1], mte[0]) * mte[4] + in_2.at<cv::Vec3b>(mte[3], mte[2]) * mte[5];
			else if constexpr (interpol_t == interpolation_type::BILINEAR)
				out.at<cv::Vec3b>(i, j) = (in_1.at<cv::Vec3b>(mte[1], mte[0]) * mte[6] + in_1.at<cv::Vec3b>(mte[1], mte[0] + 1) * mte[7] + in_1.at<cv::Vec3b>(mte[1] + 1, mte[0]) * mte[8] + in_1.at<cv::Vec3b>(mte[1] + 1, mte[0] + 1) * mte[9]) * mte[4] + (in_2.at<cv::Vec3b>(mte[3], mte[2]) * mte[10] + in_2.at<cv::Vec3b>(mte[3], mte[2] + 1) * mte[11] + in_2.at<cv::Vec3b>(mte[3] + 1, mte[2]) * mte[12] + in_2.at<cv::Vec3b>(mte[3] + 1, mte[2] + 1) * mte[13]) * mte[5];
		}
	}
	#endif
	#endif
}

/**
 * @brief Remaps input to output based on mapping table
 * 
 * @tparam interpol_t interpolation type to use
 * @param in input image
 * @param map mapping table
 * @param out output image
 */
template <interpolation_type interpol_t>
void remap(const cv::Mat &in, const cv::Mat &map, cv::Mat &out, extra_data &extra_data)
{
	#ifdef WITH_CUDA
	if constexpr (interpol_t == interpolation_type::NEAREST_NEIGHBOUR)
		cuda_remap_nn(in.data, out.data);
	else if constexpr (interpol_t == interpolation_type::BILINEAR)
		cuda_remap_li(in.data, out.data);
	#else
	#ifdef WITH_OPENCL
	CL_ERROR_CHECK( extra_data.cmdq.enqueueWriteBuffer(extra_data.in_1, CL_TRUE, 0, in.dataend - in.datastart, in.data) );
	CL_ERROR_CHECK( extra_data.cmdq.enqueueNDRangeKernel(extra_data.k, 0, extra_data.global_size, extra_data.local_size) );
	CL_ERROR_CHECK( extra_data.cmdq.finish() );
	CL_ERROR_CHECK( extra_data.cmdq.enqueueReadBuffer(extra_data.out, CL_TRUE, 0, out.dataend - out.datastart, out.data) );
	#else
	remap<interpol_t>(in(cv::Rect(0, 0, in.rows, in.rows)), in(cv::Rect(in.rows, 0, in.rows, in.rows)), map, out, extra_data);
	#endif
	#endif
}

template <bool single_input>
cv::Mat get_sharpest_image(cv::VideoCapture &in_1, cv::VideoCapture &in_2, const cv::Mat &map, extra_data &extra_data, unsigned short before = 10, unsigned short after = 10)
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
			remap<interpolation_type::NEAREST_NEIGHBOUR>(frame_1, map, eqr_frame, extra_data);
		}
		else
		{
			in_2 >> frame_2;
			remap<interpolation_type::NEAREST_NEIGHBOUR>(frame_1, frame_2, map, eqr_frame, extra_data);
		}
		cv::cvtColor(eqr_frame, grey, cv::COLOR_BGR2GRAY);
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

template <interpolation_type interpol_t, bool single_input, program_mode prog_m>
void process_video(cv::VideoCapture &in_1, cv::VideoCapture &in_2, const cv::Mat &map, cv::Mat &equiframe, cv::VideoWriter &wrt, const program_args &args, extra_data &extra_data)
{
	cv::Mat frame_1, frame_2, grey, laplace, mean, stddev, best_frame;
	double best_stddev = 0;
	int key, out_idx = 1;
	size_t search_count = args.search_range / 2;
	char full_out_path[args.out_path.size() + args.out_extension.size() + args.out_dec_len];
	bool playing = prog_m != program_mode::INTERACTIVE, play_once = true;
	TIMES(
		unsigned int nrframes = 0; double mapping_sum = 0, loading_sum = 0, preview_sum = 0;
		std::chrono::steady_clock::time_point mapping_start, mapping_end, loading_start, loading_end, preview_start, preview_end;)
	while ((key = cv::waitKey(1)) != 27)
	{
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
				if (frame_1.empty() || frame_2.empty())
					break;
			}
			TIMES(loading_end = std::chrono::steady_clock::now();
				  loading_sum += std::chrono::duration_cast<std::chrono::duration<double>>(loading_end - loading_start).count();)

			TIMES(mapping_start = std::chrono::steady_clock::now();)
			if constexpr (single_input)
				remap<interpol_t>(frame_1, map, equiframe, extra_data);
			else
				remap<interpol_t>(frame_1, frame_2, map, equiframe, extra_data);
			TIMES(mapping_end = std::chrono::steady_clock::now();
				  mapping_sum += std::chrono::duration_cast<std::chrono::duration<double>>(mapping_end - mapping_start).count();)

			TIMES(preview_start = std::chrono::steady_clock::now();)
			// Auf den Schirm Spoki
			imshow(WINDOW_NAME, equiframe);
			TIMES(preview_end = std::chrono::steady_clock::now();
				  preview_sum += std::chrono::duration_cast<std::chrono::duration<double>>(preview_end - preview_start).count();)

			// check time
			TIMES(nrframes++;
				  if ((nrframes % 150) == 0) {
					  printf("frame nr: %u\n", nrframes);
					  printf("%fms avg mapping time\n", mapping_sum / nrframes * 1000);
					  printf("%fms avg loading time\n", loading_sum / nrframes * 1000);
					  printf("%fms avg preview time\n", preview_sum / nrframes * 1000);
					  printf("%f fps mapping\n", nrframes / mapping_sum);
					  printf("%f fps mapping + loading\n", nrframes / (mapping_sum + loading_sum));
					  printf("%f fps mapping + loading + preview\n", nrframes / (mapping_sum + loading_sum + preview_sum));
					  printf("\n");
				  })
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
					equiframe = get_sharpest_image<single_input>(in_1, in_2, map, extra_data, args.search_range, args.search_range);
				imshow(WINDOW_NAME, equiframe);
				break;
			case 's':
				sprintf(full_out_path, (args.out_path + args.out_extension).c_str(), out_idx);
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
			if (args.search_range > 0)
			{
				cv::cvtColor(equiframe, grey, cv::COLOR_BGR2GRAY);
				cv::Laplacian(grey, laplace, CV_64F);
				cv::meanStdDev(laplace, mean, stddev);
				DBG(cout << "stddev: " << stddev << endl;)
				if (stddev.at<double>(0) > best_stddev)
				{
					best_frame = equiframe;
					best_stddev = stddev.at<double>(0);
				}
				search_count++;
			}
			else
			{
				best_frame = equiframe;
			}

			if (search_count == args.search_range)
			{
				DBG( cout << "best stddev: " << best_stddev << endl; )
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
				if constexpr (prog_m == program_mode::VIDEO)
				{
					wrt << best_frame;
				}
				if constexpr (prog_m == program_mode::IMAGE)
				{
					sprintf(full_out_path, (args.out_path + args.out_extension).c_str(), out_idx);
					cv::imwrite(full_out_path, best_frame);
					out_idx++;
				}
				search_count = 0;
				best_stddev = 0;
			}
		}
	}
}

template <interpolation_type interpol_t>
void process_input(program_args &args)
{
	cv::Mat frame_1, frame_2, equiframe, mapping_table;
	cv::VideoCapture cap_1, cap_2;
	cv::VideoWriter wrt;

	extra_data extra_data;
	#ifdef WITH_OPENCL
	extra_data.ctxt = cl::Context(CL_DEVICE_TYPE_GPU);
	extra_data.cmdq = cl::CommandQueue(extra_data.ctxt);
	extra_data.prog = cl::Program(extra_data.ctxt, string((char*) src_mapping_cl, src_mapping_cl_len), true);
	if constexpr (interpol_t == interpolation_type::NEAREST_NEIGHBOUR)
		extra_data.k = cl::Kernel(extra_data.prog, "remap_nn");
	else if constexpr (interpol_t == interpolation_type::BILINEAR)
		extra_data.k = cl::Kernel(extra_data.prog, "remap_li");
	#endif

	// create windows
	cv::namedWindow(WINDOW_NAME, cv::WINDOW_KEEPRATIO | cv::WINDOW_NORMAL);
	cv::resizeWindow(WINDOW_NAME, 1440, 720);

	if (args.in_path_2.empty()) // one input
	{
		read_mapping_file<interpol_t, true>(args.map_path, mapping_table);

		#ifdef WITH_CUDA
		// allocate device memory and copy mapping table
		init_device_memory(mapping_table.data, mapping_table.cols, mapping_table.rows);
		#endif
		#ifdef WITH_OPENCL
		size_t elelen = mapping_table.rows * mapping_table.cols;
		extra_data.global_size = cl::NDRange(elelen % extra_data::LOCAL_SIZE == 0 ? elelen : elelen + extra_data::LOCAL_SIZE - elelen % extra_data::LOCAL_SIZE);
		extra_data.map = cl::Buffer(extra_data.ctxt, CL_MEM_READ_WRITE, mapping_table.dataend - mapping_table.datastart);
		extra_data.in_1 = cl::Buffer(extra_data.ctxt, CL_MEM_READ_WRITE, elelen * 3);
		extra_data.out = cl::Buffer(extra_data.ctxt, CL_MEM_READ_WRITE, elelen * 3);
		CL_ERROR_CHECK( extra_data.k.setArg(0, extra_data.in_1) );
		CL_ERROR_CHECK( extra_data.k.setArg(1, extra_data.in_1) );
		CL_ERROR_CHECK( extra_data.k.setArg(2, extra_data.out) );
		CL_ERROR_CHECK( extra_data.k.setArg(3, extra_data.map) );
		CL_ERROR_CHECK( extra_data.k.setArg(4, (unsigned int) elelen) );
		CL_ERROR_CHECK( extra_data.cmdq.enqueueWriteBuffer(extra_data.map, CL_TRUE, 0, mapping_table.dataend - mapping_table.datastart, mapping_table.data) );
		#endif

		// init output mat
		equiframe.create(mapping_table.rows, mapping_table.cols, CV_8UC3);

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

		// check resolution
		if (frame_1.cols != mapping_table.cols || frame_1.rows != mapping_table.rows)
		{
			cout << "Image(" << frame_1.cols << 'x' << frame_1.rows
				 << ") and map(" << mapping_table.cols << 'x' << mapping_table.rows << ") resolution don't match." << endl;
			exit(EXIT_FAILURE);
		}

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
			string s_num_frames = to_string(args.num_frames);
			args.out_dec_len = s_num_frames.size();
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
				wrt.open(args.out_path+args.out_extension, args.out_codec == 0 ? cap_1.get(cv::CAP_PROP_FOURCC) : args.out_codec, cap_1.get(cv::CAP_PROP_FPS), cv::Size(mapping_table.cols, mapping_table.rows));
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
		read_mapping_file<interpol_t, false>(args.map_path, mapping_table);

		#ifdef WITH_CUDA
		// allocate device memory and copy mapping table
		init_device_memory(mapping_table.data, mapping_table.cols, mapping_table.rows);
		#endif
		#ifdef WITH_OPENCL
		size_t elelen = mapping_table.rows * mapping_table.cols;
		extra_data.global_size = cl::NDRange(elelen % extra_data::LOCAL_SIZE == 0 ? elelen : elelen + extra_data::LOCAL_SIZE - elelen % extra_data::LOCAL_SIZE);
		extra_data.map = cl::Buffer(extra_data.ctxt, CL_MEM_READ_WRITE, mapping_table.dataend - mapping_table.datastart);
		extra_data.in_1 = cl::Buffer(extra_data.ctxt, CL_MEM_READ_WRITE, elelen * 3);
		extra_data.in_2 = cl::Buffer(extra_data.ctxt, CL_MEM_READ_WRITE, elelen * 3);
		extra_data.out = cl::Buffer(extra_data.ctxt, CL_MEM_READ_WRITE, elelen * 3);
		CL_ERROR_CHECK( extra_data.k.setArg(0, extra_data.in_1) );
		CL_ERROR_CHECK( extra_data.k.setArg(1, extra_data.in_2) );
		CL_ERROR_CHECK( extra_data.k.setArg(2, extra_data.out) );
		CL_ERROR_CHECK( extra_data.k.setArg(3, extra_data.map) );
		CL_ERROR_CHECK( extra_data.k.setArg(4, (unsigned int) elelen) );
		CL_ERROR_CHECK( extra_data.cmdq.enqueueWriteBuffer(extra_data.map, CL_TRUE, 0, mapping_table.dataend - mapping_table.datastart, mapping_table.data) );
		#endif

		// init output mat
		equiframe.create(mapping_table.rows, mapping_table.cols, CV_8UC3);

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

		// check resolution
		if (frame_1.cols * 2 != mapping_table.cols || frame_1.rows != mapping_table.rows || frame_2.cols * 2 != mapping_table.cols || frame_2.rows != mapping_table.rows)
		{
			cout << "Image1(" << frame_1.cols << 'x' << frame_1.rows << "), Image2(" << frame_2.cols << 'x' << frame_2.rows
				 << ") and map(" << mapping_table.cols << 'x' << mapping_table.rows << ") resolution don't match." << endl;
			exit(EXIT_FAILURE);
		}

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

		// check length
		if (cap_1.get(cv::CAP_PROP_FRAME_COUNT) != cap_2.get(cv::CAP_PROP_FRAME_COUNT))
		{
			cerr << "Input 1 and 2 differ in length." << endl;
			exit(EXIT_FAILURE);
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
			string s_num_frames = to_string(args.num_frames);
			args.out_dec_len = s_num_frames.size();
			args.out_path += "%0" + to_string(args.out_dec_len) + 'd';
		}
		
		// jump back to beginning
		cap_1.set(cv::CAP_PROP_POS_FRAMES, 0);
		cap_2.set(cv::CAP_PROP_POS_FRAMES, 0);

		// do mapping
		switch (args.prog_mode)
		{
			case program_mode::INTERACTIVE:
				process_video<interpol_t, false, program_mode::INTERACTIVE>(cap_1, cap_2, mapping_table, equiframe, wrt, args, extra_data);
				break;
			case program_mode::VIDEO:
				wrt.open(args.out_path+args.out_extension, args.out_codec == 0 ? cap_1.get(cv::CAP_PROP_FOURCC) : args.out_codec, cap_1.get(cv::CAP_PROP_FPS), cv::Size(mapping_table.cols, mapping_table.rows));
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
				"  f\t\tget sharpest image in range of +-10 frames\n  esc\t\texit" << endl;
	}
	else
	{
		cout << "controls:\n  esc\t\texit" << endl;
	}

	switch (args.interpol_t)
	{
	case interpolation_type::NEAREST_NEIGHBOUR:
		process_input<interpolation_type::NEAREST_NEIGHBOUR>(args);
		break;
	case interpolation_type::BILINEAR:
		process_input<interpolation_type::BILINEAR>(args);
		break;
	default:
		cerr << "Interpolation type not supported." << endl;
		break;
	}

	return EXIT_SUCCESS;
}

template <bool single_input>
void mapper::init(interpolation_type interpol_t, size_t buffer_length)
{
	this->buffer_length = buffer_length;
	this->read_pos = buffer_length;
	this->write_pos = 0;

	this->unread = new atomic_bool[buffer_length + 1];
	this->unwritten = new atomic_bool[buffer_length];
	this->out = new cv::Mat[buffer_length];

	for (size_t i = 0; i < buffer_length; i++)
	{
		this->out[i].create(this->map.rows, this->map.cols, CV_8UC3);
		this->unwritten[i] = true;
		this->unread[i] = false;
	}
	
	this->alive = true;
	this->running = true;
	this->good = true;

	#ifdef WITH_OPENCL
	this->edata.ctxt = cl::Context(CL_DEVICE_TYPE_GPU);
	this->edata.cmdq = cl::CommandQueue(this->edata.ctxt);
	this->edata.prog = cl::Program(this->edata.ctxt, string((char*) src_mapping_cl, src_mapping_cl_len), true);
	
	size_t elelen = this->map.rows * this->map.cols;
	this->edata.global_size = cl::NDRange(elelen % extra_data::LOCAL_SIZE == 0 ? elelen : elelen + extra_data::LOCAL_SIZE - elelen % extra_data::LOCAL_SIZE);
	this->edata.map = cl::Buffer(this->edata.ctxt, CL_MEM_READ_WRITE, this->map.dataend - this->map.datastart);
	if constexpr (single_input)
		this->edata.in_1 = cl::Buffer(this->edata.ctxt, CL_MEM_READ_WRITE, elelen * 3);
	else
	{
		this->edata.in_1 = cl::Buffer(this->edata.ctxt, CL_MEM_READ_WRITE, elelen * 3 / 2);
		this->edata.in_2 = cl::Buffer(this->edata.ctxt, CL_MEM_READ_WRITE, elelen * 3 / 2);
	}
	this->edata.out = cl::Buffer(this->edata.ctxt, CL_MEM_READ_WRITE, elelen * 3);
	switch (interpol_t)
	{
	case interpolation_type::NEAREST_NEIGHBOUR:
		this->edata.k = cl::Kernel(this->edata.prog, "remap_nn");
		break;
	case interpolation_type::BILINEAR:
		this->edata.k = cl::Kernel(this->edata.prog, "remap_li");
		break;
	default:
		throw invalid_argument("Interpolation type not supported.");
		break;
	}
	CL_ERROR_CHECK( this->edata.k.setArg(0, this->edata.in_1) );
	if constexpr (single_input)
	{
		CL_ERROR_CHECK( this->edata.k.setArg(1, this->edata.in_1) );
	}
	else
	{
		CL_ERROR_CHECK( this->edata.k.setArg(1, this->edata.in_2) );
	}
	CL_ERROR_CHECK( this->edata.k.setArg(2, this->edata.out) );
	CL_ERROR_CHECK( this->edata.k.setArg(3, this->edata.map) );
	CL_ERROR_CHECK( this->edata.k.setArg(4, (unsigned int) elelen) );
	CL_ERROR_CHECK( this->edata.cmdq.enqueueWriteBuffer(this->edata.map, CL_TRUE, 0, this->map.dataend - this->map.datastart, this->map.data) );
	#endif

	switch (interpol_t)
	{
	case interpolation_type::NEAREST_NEIGHBOUR:
		this->worker = std::thread(&mapper::process<interpolation_type::NEAREST_NEIGHBOUR, single_input>, this);
		break;
	case interpolation_type::BILINEAR:
		this->worker = std::thread(&mapper::process<interpolation_type::BILINEAR, single_input>, this);
		break;
	default:
		throw invalid_argument("Interpolation type not supported.");
		break;
	}
}

mapper::mapper(cv::VideoCapture &in, const cv::Mat map, size_t frameskip, interpolation_type interpol_t, size_t buffer_length)
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

mapper::mapper(cv::VideoCapture &in_1, cv::VideoCapture in_2, const cv::Mat map, size_t frameskip, interpolation_type interpol_t, size_t buffer_length)
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

mapper::mapper(const std::string &in_path, const std::string &map_path, size_t frameskip, interpolation_type interpol_t, size_t buffer_length)
{
	if (!this->cap_1.open(in_path))
		throw invalid_argument('"' + in_path + "\" is can't be opened for reading.");

	switch (interpol_t)
	{
	case interpolation_type::NEAREST_NEIGHBOUR:
		read_mapping_file<interpolation_type::NEAREST_NEIGHBOUR, true>(map_path, this->map);
		break;
	case interpolation_type::BILINEAR:
		read_mapping_file<interpolation_type::BILINEAR, true>(map_path, this->map);
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

mapper::mapper(const std::string &in_path_1, const std::string &in_path_2, const std::string &map_path, size_t frameskip, interpolation_type interpol_t, size_t buffer_length)
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
	case interpolation_type::NEAREST_NEIGHBOUR:
		read_mapping_file<interpolation_type::NEAREST_NEIGHBOUR, false>(map_path, this->map);
		break;
	case interpolation_type::BILINEAR:
		read_mapping_file<interpolation_type::BILINEAR, false>(map_path, this->map);
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

template <interpolation_type interpol_t, bool single_input>
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
	DBG( printf("getimg: r%lu unlocked\n", this->read_pos); )
	if (++this->read_pos >= this->buffer_length)
		this->read_pos = 0;

	bool lk;
	while ((lk = this->unwritten[this->read_pos].exchange(true)) && this->good)
		this_thread::sleep_for(1ms);
	if (!lk)
	{
		DBG( printf("getimg: w%lu locked\n", this->read_pos); )
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
