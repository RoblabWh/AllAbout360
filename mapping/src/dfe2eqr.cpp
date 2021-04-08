#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "mapping.h"

// g++ -O2 -fopenmp -o dfe2eqr dfe2eqr.cpp  `pkg-config --cflags opencv` `pkg-config --libs opencv`

using namespace std;

const string WINDOW_NAME = "Output";

struct program_args
{
	string in_path_1, in_path_2, map_path, out_path, out_extension;
	interpolation_type interpol_t = interpolation_type::NEAREST_NEIGHBOUR;
	bool vid_out = false;
	int out_codec;
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
	DBG(cout << "mappingtable size:" << width << 'x' << height << endl;)
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
				#ifdef CUDA
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
							 + argv[0] + string(" [-vo] [-li] [-c <codec>] [-o <output file>] -m <map file> <input file>\n  ")
							 + argv[0] + string(" [-vo] [-li] [-c <codec>] [-o <output file>] -m <map file> <input file 1> <input file 2>");
	const string HELP_TEXT = "options:\n"
							 "  --help\t\t\t-h\tprint this\n"
							 "  --mappingfile\t\t\t-m\tmapping file path\n"
							 "  --output\t\t\t-o\toutput file path\n"
							 "  --linear-interpolation\t-li\tenable linear interpolation\n"
							 "  --video\t\t\t\tconvert to video\n"
							 "  --codec\t--fourcc\t\tcodec for video output";

	int i;
	// parse args
	for (i = 1; i < argc; i++)
	{
		if (string("--help") == argv[i] || string("-h") == argv[i])
		{
			cout << HELP_TEXT << '\n' << USAGE_TEXT << endl;
			exit(EXIT_SUCCESS);
		}
		else if (string("--output") == argv[i] || string("-o") == argv[i])
		{
			i++;
			if (i < argc)
			{
				args.out_path = argv[i];
				size_t ext_begin = args.out_path.find_last_of('.');
				args.out_extension = args.out_path.substr(ext_begin);
				args.out_path = args.out_path.substr(0, args.out_path.size() - ext_begin);
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
		else if (string("--video") == argv[i])
		{
			args.vid_out = true;
		}
		else if (string("--codec") == argv[i] || string("--fourcc") == argv[i])
		{
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
		else break;
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
		if (args.vid_out)
			args.out_extension = ".mkv";
		else
			args.out_extension = ".png";
	}
	
	// parse input files
	if (argc - i == 1)
	{
		args.in_path_1 = argv[i];
		cout << "Processing video file \"" << args.in_path_1 << "\" with mapping table \"" << args.map_path << "\"." << endl;
	}
	else if (argc - i == 2)
	{
		args.in_path_1 = argv[i];
		args.in_path_2 = argv[i + 1];
		cout << "Processing video files \"" << args.in_path_1 << "\" and \"" << args.in_path_2 << "\" with mapping table \"" << args.map_path << "\"." << endl;
	}
	else
	{
		cerr << "too many arguments." << endl;
		exit(EXIT_FAILURE);
	}
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
void remap(const cv::Mat &in_1, const cv::Mat &in_2, const cv::Mat &map, cv::Mat &out)
{
	#ifdef CUDA
	if constexpr (interpol_t == interpolation_type::NEAREST_NEIGHBOUR)
		cuda_remap_nn(in_1.data, in_2.data, out.data);
	else if constexpr (interpol_t == interpolation_type::BILINEAR)
		cuda_remap_li(in_1.data, in_2.data, out.data);
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
void remap(const cv::Mat &in, const cv::Mat &map, cv::Mat &out)
{
	#ifdef CUDA
	if constexpr (interpol_t == interpolation_type::NEAREST_NEIGHBOUR)
		cuda_remap_nn(in.data, out.data);
	else if constexpr (interpol_t == interpolation_type::BILINEAR)
		cuda_remap_li(in.data, out.data);
	#else
	remap<interpol_t>(in(cv::Rect(0, 0, in.rows, in.rows)), in(cv::Rect(in.rows, 0, in.rows, in.rows)), map, out);
	#endif
}

template <bool single_input>
cv::Mat get_sharpest_image(cv::VideoCapture &in_1, cv::VideoCapture &in_2, const cv::Mat &map, unsigned short before = 10, unsigned short after = 10)
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
			remap<interpolation_type::NEAREST_NEIGHBOUR>(frame_1, map, eqr_frame);
		}
		else
		{
			in_2 >> frame_2;
			remap<interpolation_type::NEAREST_NEIGHBOUR>(frame_1, frame_2, map, eqr_frame);
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

template <interpolation_type interpol_t, bool single_input, bool wrt_out>
void process_video(cv::VideoCapture &in_1, cv::VideoCapture &in_2, const cv::Mat &map, cv::Mat &equiframe, cv::VideoWriter &wrt, const program_args &args)
{
	cv::Mat frame_1, frame_2;
	int key, out_idx = 1;
	bool playing = wrt_out, play_once = true;
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
				remap<interpol_t>(frame_1, map, equiframe);
			else
				remap<interpol_t>(frame_1, frame_2, map, equiframe);
			TIMES(mapping_end = std::chrono::steady_clock::now();
				  mapping_sum += std::chrono::duration_cast<std::chrono::duration<double>>(mapping_end - mapping_start).count();)

			TIMES(preview_start = std::chrono::steady_clock::now();)
			// Auf den Schirm Spoki
			imshow(WINDOW_NAME, equiframe);
			TIMES(preview_end = std::chrono::steady_clock::now();
				  preview_sum += std::chrono::duration_cast<std::chrono::duration<double>>(preview_end - preview_start).count();)

			// check time
			TIMES(nrframes++;
				  if ((nrframes % 300) == 0) {
					  printf("frame nr: %u\n", nrframes);
					  printf("%f avg mapping time\n", mapping_sum / nrframes);
					  printf("%f avg loading time\n", loading_sum / nrframes);
					  printf("%f avg preview time\n", preview_sum / nrframes);
					  printf("%f fps mapping\n", nrframes / mapping_sum);
					  printf("%f fps mapping + loading\n", nrframes / (mapping_sum + loading_sum));
					  printf("%f fps mapping + loading + preview\n", nrframes / (mapping_sum + loading_sum + preview_sum));
					  printf("\n");
				  })
			play_once = false;
		}
		if constexpr (wrt_out)
		{
			wrt << equiframe;
		}
		else
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
				equiframe = get_sharpest_image<single_input>(in_1, in_2, map);
				imshow(WINDOW_NAME, equiframe);
				break;
			case 's':
				cv::imwrite(args.out_path + '_' + to_string(out_idx) + args.out_extension, equiframe);
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
	}
}

template <interpolation_type interpol_t>
void process_input(const program_args &args)
{
	cv::Mat frame_1, frame_2, equiframe, mapping_table;
	cv::VideoCapture cap_1, cap_2;
	cv::VideoWriter wrt;

	// create windows
	cv::namedWindow(WINDOW_NAME, cv::WINDOW_KEEPRATIO | cv::WINDOW_NORMAL);
	cv::resizeWindow(WINDOW_NAME, 1440, 720);

	if (args.in_path_2.empty()) // one input
	{
		read_mapping_file<interpol_t, true>(args.map_path, mapping_table);

		#ifdef CUDA
		// allocate device memory and copy mapping table
		init_device_memory(mapping_table.data, mapping_table.cols, mapping_table.rows);
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
			remap<interpol_t>(frame_1, mapping_table, equiframe);
			cv::imshow(WINDOW_NAME, equiframe);
			if (!args.out_path.empty())
				cv::imwrite(args.out_path+args.out_extension, equiframe);
			cv::waitKey();
			return;
		}

		// jump back to beginning
		cap_1.set(cv::CAP_PROP_POS_FRAMES, 0);

		// do mapping
		if (args.vid_out)
		{
			wrt.open(args.out_path+args.out_extension, args.out_codec == 0 ? cap_1.get(cv::CAP_PROP_FOURCC) : args.out_codec, cap_1.get(cv::CAP_PROP_FPS), cv::Size(mapping_table.cols, mapping_table.rows));
			process_video<interpol_t, true, true>(cap_1, cap_2, mapping_table, equiframe, wrt, args);
		}
		else
		{
			process_video<interpol_t, true, false>(cap_1, cap_2, mapping_table, equiframe, wrt, args);
		}
	}
	else // two inputs
	{
		read_mapping_file<interpol_t, false>(args.map_path, mapping_table);

		#ifdef CUDA
		// allocate device memory and copy mapping table
		init_device_memory(mapping_table.data, mapping_table.cols, mapping_table.rows);
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
			remap<interpol_t>(frame_1, frame_2, mapping_table, equiframe);
			cv::imshow(WINDOW_NAME, equiframe);
			if (!args.out_path.empty())
				cv::imwrite(args.out_path+args.out_extension, equiframe);
			cv::waitKey();
			return;
		}

		// jump back to beginning
		cap_1.set(cv::CAP_PROP_POS_FRAMES, 0);
		cap_2.set(cv::CAP_PROP_POS_FRAMES, 0);

		// do mapping
		if (args.vid_out)
		{
			wrt.open(args.out_path+args.out_extension, args.out_codec == 0 ? cap_1.get(cv::CAP_PROP_FOURCC) : args.out_codec, cap_1.get(cv::CAP_PROP_FPS), cv::Size(mapping_table.cols, mapping_table.rows));
			process_video<interpol_t, false, true>(cap_1, cap_2, mapping_table, equiframe, wrt, args);
		}
		else
		{
			process_video<interpol_t, false, false>(cap_1, cap_2, mapping_table, equiframe, wrt, args);
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

	if (!args.vid_out)
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
