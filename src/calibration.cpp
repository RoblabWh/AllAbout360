#include "iostream"
#include "cmath"
#include "fstream"
#include "string"

#include <opencv2/opencv.hpp>

#include "mapping.h"

#ifdef WITH_OPENCL
#include <CL/cl.hpp>
#include "clerror.h"
#include "mapping.cl.h"
#endif

using namespace std;
using namespace cv;

// Type of mapping table entry
typedef cv::Vec<double, 14> mte_t;

const int FOV_DEFAULT = 190; // >= 180

const int SLIDER_PIXEL_RANGE = 100, SLIDER_PIXEL_DEFAULT = SLIDER_PIXEL_RANGE / 2;
const int SLIDER_FACTOR_RANGE = 200, SLIDER_FACTOR_DEFAULT = SLIDER_FACTOR_RANGE / 2;
const int SLIDER_ROTATION_RANGE = 3600, SLIDER_ROTATION_DEFAULT = SLIDER_ROTATION_RANGE / 2;
const int SLIDER_ORIENTATION_RANGE = 360, SLIDER_ORIENTATION_DEFAULT = SLIDER_ORIENTATION_RANGE / 2;
const int SLIDER_FOV_RANGE = 100, SLIDER_FOV_DEFAULT = 10;
const int SLIDER_BLEND_RANGE = SLIDER_FOV_RANGE / 2, SLIDER_BLEND_DEFAULT = 0;
const string SLIDER_PIXEL_NAME = string(" += slider - ") + to_string(SLIDER_PIXEL_DEFAULT);
const string SLIDER_FACTOR_NAME = string(" *= slider / ") + to_string(SLIDER_FACTOR_DEFAULT);
const string SLIDER_X_OFF_NAME = string("x") + SLIDER_PIXEL_NAME;
const string SLIDER_Y_OFF_NAME = string("y") + SLIDER_PIXEL_NAME;
const string SLIDER_RADIUS_NAME = string("radius") + SLIDER_FACTOR_NAME;
const string SLIDER_ROTATION_NAME = string("rotation = ((slider - ") + to_string(SLIDER_ROTATION_DEFAULT) + ") / 10.0)°";
const string SLIDER_ORIENTATION_NAME = string("orientation = (slider - ") + to_string(SLIDER_ORIENTATION_DEFAULT) + ")°";
const string SLIDER_FOV_NAME = string("fov = (") + to_string(FOV_DEFAULT) + string(" + slider - ") + to_string(SLIDER_FOV_DEFAULT) + ")°";
const string SLIDER_BLEND_NAME = string("blend size = (slider * 2)°");

const char CAMERA_FRONT_WINDOW_NAME[] = "Camera front";
const char CAMERA_REAR_WINDOW_NAME[] = "Camera rear";
const char PREVIEW_WINDOW_NAME[] = "Preview";
const char HELP_DIALOG[] = " (<file> | <file_front> <file_rear>) [--resolution -r <width> <height>] [--fps -f <fps>] [--codec -c <fourcc>]"
						   " [--parameters -p <x_offset_front> <y_offset_front> <rotation_front> <fov_front>"
						   " <radius_front> <x_offset_rear> <y_offset_rear> <rotation_rear> <radius_rear> <fov_rear>] [--map -m | --opencv -cv | --integer -i]\n\n"
						   "Options:\n"
						   "  --output      -o   mapping table output file\n"
						   "  --parameters  -p   set the inital slider parameters\n"
						   "  --map         -m   output in full format double[x1, y1, x2, y2, blend factor]\n"
						   "  --opencv      -cv  output in OpenCV remap format double[x, y]\n"
						   "  --integer     -i   output in integer format int[x, y]\n"
						   " For usage with cameras: (WARNING! Using OpenCV VideoCapture properties, which might not work correctly)\n"
						   "  --resolution  -r   set the camera resolution\n"
						   "  --fps         -f   set the camera framerate\n"
						   "  --codec       -c   set the camera codec in fourcc format\n";

enum output_format
{
	PARAMETER, // double x_off_front, y_off_front, radius_front, rot_front, x_off_rear, y_off_rear, radius_rear, rot_rear, front_limit, blend_limit, blend_size
	FULL, // double[x1, y1, x2, y2, blend factor]
	OPENCV, // double[x, y]
	INTEGER // int[x, y]
};

/**
 * @brief Holds slider parameters
 */
struct calib_slider_params
{
	int x_off_front, y_off_front, rot_front, radius_front, fov_front,
		x_off_rear, y_off_rear, rot_rear, radius_rear, fov_rear,
		rot_both, blend_size, orientation;
};

/**
 * @brief Hold the arguments parsed from commandline
 */
struct program_args
{
	char *file, *file_front, *file_rear, *fourcc_str;
	string output_file;
	int width, height, fps, fourcc;
	calib_slider_params *params;
	bool is_single_input, has_resolution, has_fps, has_fourcc, has_parameters, has_output;
	output_format format;
};

/**
 * @brief Converts degrees to radians.
 * @param deg degrees
 * @return radians
 */
double inline deg2rad(double deg) { return deg * M_PI / 180.; }

/**
 * @brief Parses the arguments given in commandline
 *
 * @param argc argc from main
 * @param argv argv from main
 * @param args parsed arguments
 */
void parse_args(int argc, char **argv, program_args *args)
{
	if (argc == 1)
	{
		cout << argv[0] << HELP_DIALOG << endl;
		exit(1);
	}
	if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
	{
		cout << argv[0] << HELP_DIALOG << endl;
		exit(0);
	}

	int i = 2;
	if (argc >= 3 && argv[2][0] != '-')
	{
		args->file_front = argv[1];
		args->file_rear = argv[2];
		args->is_single_input = false;
		i++;
	}
	else
	{
		args->file = argv[1];
		args->is_single_input = true;
	}

	for (; i < argc; i++)
	{
		if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--resolution") == 0)
		{
			if (i >= argc - 2)
			{
				cout << "No resolution provided." << endl;
				exit(1);
			}
			args->width = stoi(argv[i + 1]);
			args->height = stoi(argv[i + 2]);
			args->has_resolution = true;
			i += 2;
		}
		else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--fps") == 0)
		{
			i++;
			if (i >= argc)
			{
				cout << "No framerate provided." << endl;
				exit(1);
			}
			args->fps = stoi(argv[i]);
			args->has_fps = true;
		}
		else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--codec") == 0)
		{
			i++;
			if (i >= argc)
			{
				cout << "No codec provided." << endl;
				exit(1);
			}
			else if (strlen(argv[i]) != 4)
			{
				cout << "Codec needs to be exactly four characters long." << endl;
				exit(1);
			}
			args->fourcc = cv::VideoWriter::fourcc(argv[i][0], argv[i][1], argv[i][2], argv[i][3]);
			args->fourcc_str = argv[i];
			args->has_fourcc = true;
		}
		else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--parameters") == 0)
		{
			if (i >= argc - 11)
			{
				cout << "All paramters have to be given." << endl;
				exit(1);
			}
			args->params->x_off_front = stoi(argv[i + 1]);
			args->params->y_off_front = stoi(argv[i + 2]);
			args->params->rot_front = stoi(argv[i + 3]);
			args->params->radius_front = stoi(argv[i + 4]);
			args->params->fov_front = stoi(argv[i + 5]);

			args->params->x_off_rear = stoi(argv[i + 6]);
			args->params->y_off_rear = stoi(argv[i + 7]);
			args->params->rot_rear = stoi(argv[i + 8]);
			args->params->radius_rear = stoi(argv[i + 9]);
			args->params->fov_rear = stoi(argv[i + 10]);

			args->params->blend_size = stoi(argv[i + 11]);

			args->has_parameters = true;
			i += 11;
		}
		else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--map") == 0)
		{
			if (args->format != output_format::PARAMETER)
			{
				cout << "Only one output format can be set at a time." << endl;
				exit(1);
			}
			args->format = output_format::FULL;
		}
		else if (strcmp(argv[i], "-cv") == 0 || strcmp(argv[i], "--opencv") == 0)
		{
			if (args->format != output_format::PARAMETER)
			{
				cout << "Only one output format can be set at a time." << endl;
				exit(1);
			}
			args->format = output_format::OPENCV;
		}
		else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--integer") == 0)
		{
			if (args->format != output_format::PARAMETER)
			{
				cout << "Only one output format can be set at a time." << endl;
				exit(1);
			}
			args->format = output_format::INTEGER;
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			cout << argv[0] << HELP_DIALOG << endl;
			exit(0);
		}
		else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0)
		{
			i++;
			if (i >= argc)
			{
				cout << "No output provided." << endl;
				exit(1);
			}
			args->output_file = argv[i];
		}
		else
		{
			cout << "unknown argument \"" << argv[i] << "\"" << endl;
			exit(1);
		}
	}

	// set output file name if none is given
	if (args->output_file.empty())
	{
		args->has_output = false;
		if (args->format == PARAMETER)
			args->output_file = "mapping-parameter.txt";
		else
			args->output_file = "mapping-table.txt";
	}
	else
		args->has_output = true;
}

int main(int argc, char **argv)
{
	int i, j;
	int height, width;
	program_args args = {0};
	calib_params params = {0};
	calib_slider_params slider_params = {SLIDER_FACTOR_DEFAULT, SLIDER_FACTOR_DEFAULT, SLIDER_ROTATION_DEFAULT, SLIDER_FACTOR_DEFAULT, SLIDER_FOV_DEFAULT,
										 SLIDER_FACTOR_DEFAULT, SLIDER_FACTOR_DEFAULT, SLIDER_ROTATION_DEFAULT, SLIDER_FACTOR_DEFAULT, SLIDER_FOV_DEFAULT,
										 SLIDER_ROTATION_DEFAULT, 0, SLIDER_ORIENTATION_DEFAULT};

	args.params = &slider_params;

	// parse arguments and set optional parameters
	parse_args(argc, argv, &args);

	VideoCapture cap, cap_front, cap_rear;
	if (args.is_single_input)
	{
		// try to open video
		cap.open(args.file);
		if (!cap.isOpened())
		{
			printf("Could not open video %s\n", args.file);
			return EXIT_FAILURE;
		}

		// set optional fps for cameras
		if (args.has_fps)
		{
			if (!cap.set(CAP_PROP_FPS, args.fps))
				printf("Couldn't set FPS to %d\n", args.fps);
		}
		// TODO sets caps but cameras don't change
		// set optional codec for cameras
		if (args.has_fourcc)
		{
			cout << "fourcc: " << (int) cap.get(CAP_PROP_FOURCC) << endl;
			if (!cap.set(CAP_PROP_FOURCC, static_cast<double>(args.fourcc)))
				printf("Couldn't set codec to %s\n", args.fourcc_str);

			cout << "fourcc: " << (int) cap.get(CAP_PROP_FOURCC) << "  " << static_cast<double>(args.fourcc) << "  " << args.fourcc << "  " << args.fourcc_str << endl;
			DBG(cout << args.fourcc_str << endl);
		}
		// set optional resolution for cameras
		if (args.has_resolution)
		{
			// try to set given resolution and check success
			if (!cap.set(CAP_PROP_FRAME_WIDTH, args.width))
				printf("Couldn't set width to %d\n", args.width);
			if (!cap.set(CAP_PROP_FRAME_HEIGHT, args.height))
				printf("Couldn't set heigth to %d\n", args.height);
		}

		width = cap.get(CAP_PROP_FRAME_WIDTH);
		height = cap.get(CAP_PROP_FRAME_HEIGHT);
	}
	else
	{
		cap_front.open(args.file_front);
		cap_rear.open(args.file_rear);
		if (!cap_front.isOpened())
		{
			printf("Could not open front video %s\n", args.file_front);
			return EXIT_FAILURE;
		}
		if (!cap_rear.isOpened())
		{
			printf("Could not open rear video %s\n", args.file_rear);
			return EXIT_FAILURE;
		}
		if (cap_front.get(CAP_PROP_FRAME_WIDTH) != cap_rear.get(CAP_PROP_FRAME_WIDTH) || cap_front.get(CAP_PROP_FRAME_HEIGHT) != cap_rear.get(CAP_PROP_FRAME_HEIGHT))
		{
			printf("Resolution of the front and rear video differ.\n");
			return EXIT_FAILURE;
		}

		// set optional fps for cameras
		if (args.has_fps)
		{
			if (!cap_front.set(CAP_PROP_FPS, args.fps))
				printf("Couldn't set front FPS to %d\n", args.fps);
			if (!cap_rear.set(CAP_PROP_FPS, args.fps))
				printf("Couldn't set rear FPS to %d\n", args.fps);
		}
		// TODO sets caps but cameras don't change
		// set optional codec for cameras
		if (args.has_fourcc)
		{
			if (!cap_front.set(CAP_PROP_FOURCC, static_cast<double>(args.fourcc)))
				printf("Couldn't set front codec to %s\n", args.fourcc_str);
			if (!cap_rear.set(CAP_PROP_FOURCC, static_cast<double>(args.fourcc)))
				printf("Couldn't set rear codec to %s\n", args.fourcc_str);
		}
		// set optional resolution for cameras
		if (args.has_resolution)
		{
			// try to set given resolution and check success
			if (!cap_front.set(CAP_PROP_FRAME_WIDTH, args.width))
				printf("Couldn't set front width to %d\n", args.width);
			if (!cap_front.set(CAP_PROP_FRAME_HEIGHT, args.height))
				printf("Couldn't set front heigth to %d\n", args.height);
			if (!cap_rear.set(CAP_PROP_FRAME_WIDTH, args.width))
				printf("Couldn't set rear width to %d\n", args.width);
			if (!cap_rear.set(CAP_PROP_FRAME_HEIGHT, args.height))
				printf("Couldn't set rear heigth to %d\n", args.height);
		}

		width = cap_front.get(CAP_PROP_FRAME_WIDTH) * 2;
		height = cap_front.get(CAP_PROP_FRAME_HEIGHT);
	}

	// create windows
	namedWindow(CAMERA_FRONT_WINDOW_NAME, WINDOW_NORMAL | WINDOW_KEEPRATIO);
	namedWindow(CAMERA_REAR_WINDOW_NAME, WINDOW_NORMAL | WINDOW_KEEPRATIO);
	namedWindow(PREVIEW_WINDOW_NAME, WINDOW_NORMAL | WINDOW_KEEPRATIO);
	resizeWindow(CAMERA_FRONT_WINDOW_NAME, 720, 720);
	resizeWindow(CAMERA_REAR_WINDOW_NAME, 720, 720);
	resizeWindow(PREVIEW_WINDOW_NAME, 1440, 720);

	// create slider
	createTrackbar(SLIDER_X_OFF_NAME, CAMERA_FRONT_WINDOW_NAME, &(slider_params.x_off_front), SLIDER_FACTOR_RANGE);
	createTrackbar(SLIDER_Y_OFF_NAME, CAMERA_FRONT_WINDOW_NAME, &(slider_params.y_off_front), SLIDER_FACTOR_RANGE);
	createTrackbar(SLIDER_RADIUS_NAME, CAMERA_FRONT_WINDOW_NAME, &(slider_params.radius_front), SLIDER_FACTOR_RANGE);
	createTrackbar(SLIDER_ROTATION_NAME, CAMERA_FRONT_WINDOW_NAME, &(slider_params.rot_front), SLIDER_ROTATION_RANGE);
	// createTrackbar(SLIDER_FOV_NAME, CAMERA_FRONT_WINDOW_NAME, &(slider_params.fov_front), SLIDER_FOV_RANGE);

	createTrackbar(SLIDER_X_OFF_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.x_off_rear), SLIDER_FACTOR_RANGE);
	createTrackbar(SLIDER_Y_OFF_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.y_off_rear), SLIDER_FACTOR_RANGE);
	createTrackbar(SLIDER_RADIUS_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.radius_rear), SLIDER_FACTOR_RANGE);
	createTrackbar(SLIDER_ROTATION_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.rot_rear), SLIDER_ROTATION_RANGE);
	// createTrackbar(SLIDER_FOV_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.fov_rear), SLIDER_FOV_RANGE);

	createTrackbar(SLIDER_FOV_NAME, PREVIEW_WINDOW_NAME, &(slider_params.fov_front), SLIDER_FOV_RANGE);
	createTrackbar(SLIDER_ROTATION_NAME, PREVIEW_WINDOW_NAME, &(slider_params.rot_both), SLIDER_ROTATION_RANGE);
	createTrackbar(SLIDER_ORIENTATION_NAME, PREVIEW_WINDOW_NAME, &(slider_params.orientation), SLIDER_ORIENTATION_RANGE);
	createTrackbar(SLIDER_BLEND_NAME, PREVIEW_WINDOW_NAME, &(slider_params.blend_size), SLIDER_BLEND_RANGE);

	Mat frame, frame_front, frame_rear, equiframe, drawframe_front, drawframe_rear;
	Point center_f, center_r, line_f, line_r;
	Scalar prev_color(0, 255, 0), prev_blend_color(0, 127, 255);
	int radius_f, radius_r, radius_blend_f, radius_blend_r, ch = 0, height_2 = height / 2, line_width = width / 1000 + 1;
	bool paused = false, play_once = false, video_input = true;

	// get first image and check if the file is empty
	if (args.is_single_input)
	{
		cap >> frame;
		if (frame.empty())
		{
			printf("Video is empty.\n");
			return EXIT_FAILURE;
		}
		DBG(cout << "width: " << frame.cols << " height: " << frame.rows << endl);
		// split frame in front and rear
		frame_front = frame(Rect(0, 0, frame.rows, frame.rows));
		frame_rear = frame(Rect(frame.rows, 0, frame.rows, frame.rows));

		if (!cap.grab()) video_input = false;
	}
	else
	{
		cap_front >> frame_front;
		if (frame_front.empty())
		{
			printf("Front video is empty.\n");
			return EXIT_FAILURE;
		}
		cap_rear >> frame_rear;
		if (frame_rear.empty())
		{
			printf("Rear video is empty.\n");
			return EXIT_FAILURE;
		}
		if (!cap_front.grab()) video_input = false;
	}
	// init output image
	equiframe.create(height, width, frame_front.type());
	// equiframe.create(height, width / 4 * 3, frame_front.type());
	// create mapping table
	Mat mapping_table;

	extra_data extra_data;
	#ifdef WITH_OPENCL
	CL_ARGERR_CHECK( extra_data.ctxt = cl::Context(CL_DEVICE_TYPE_GPU, NULL, NULL, NULL, &_cl_error); )
	CL_ARGERR_CHECK( extra_data.cmdq = cl::CommandQueue(extra_data.ctxt, 0, &_cl_error); )
	CL_ARGERR_CHECK( extra_data.prog = cl::Program(extra_data.ctxt, string((char*) src_mapping_cl, src_mapping_cl_len), true, &_cl_error); )
	CL_ARGERR_CHECK( extra_data.k = cl::Kernel(extra_data.prog, "remap_nearest", &_cl_error); )
	size_t elelen = width * height;
	extra_data.global_size = cl::NDRange(elelen % extra_data::LOCAL_SIZE == 0 ? elelen : elelen + extra_data::LOCAL_SIZE - elelen % extra_data::LOCAL_SIZE);
	CL_ARGERR_CHECK( extra_data.map = cl::Buffer(extra_data.ctxt, CL_MEM_READ_WRITE, elelen * 14 * 8, NULL, &_cl_error); )
	CL_ARGERR_CHECK( extra_data.in_1 = cl::Buffer(extra_data.ctxt, CL_MEM_READ_WRITE, elelen * 3, NULL, &_cl_error); )
	if (args.is_single_input)
	{
		CL_ARGERR_CHECK( extra_data.in_1 = cl::Buffer(extra_data.ctxt, CL_MEM_READ_WRITE, elelen * 3, NULL, &_cl_error); )
	}
	else
	{
		CL_ARGERR_CHECK( extra_data.in_1 = cl::Buffer(extra_data.ctxt, CL_MEM_READ_WRITE, elelen / 2 * 3, NULL, &_cl_error); )
		CL_ARGERR_CHECK( extra_data.in_2 = cl::Buffer(extra_data.ctxt, CL_MEM_READ_WRITE, elelen / 2 * 3, NULL, &_cl_error); )
	}
	CL_ARGERR_CHECK( extra_data.out = cl::Buffer(extra_data.ctxt, CL_MEM_READ_WRITE, elelen * 3, NULL, &_cl_error); )
	CL_RETERR_CHECK( extra_data.k.setArg(0, extra_data.in_1) );
	if (args.is_single_input)
	{
		CL_RETERR_CHECK( extra_data.k.setArg(1, extra_data.in_1) );
	}
	else
	{
		CL_RETERR_CHECK( extra_data.k.setArg(1, extra_data.in_2) );
	}
	CL_RETERR_CHECK( extra_data.k.setArg(2, extra_data.out) );
	CL_RETERR_CHECK( extra_data.k.setArg(3, extra_data.map) );
	CL_RETERR_CHECK( extra_data.k.setArg(4, (unsigned int) elelen) );
	#endif

	printf("Controls:\n"
		   "  space/k:  start/stop videoplayback\n"
		   "  j/l:      jump 10 second back/forward\n"
		   "  <-/->:    decrement/increment current slider\n"
		   "  tab:      next slider\n"
		   "  enter:    save to file and exit\n"
		   "  esc:      exit without saving\n");

	TIMES(
		chrono::time_point<chrono::steady_clock> loop_s, loop_e, read_s, read_e, genm_s, genm_e, rmap_s, rmap_e, slid_s, slid_e, prev_s, prev_e;
		double loop_sum = 0, read_sum = 0, genm_sum = 0, rmap_sum = 0, slid_sum = 0, prev_sum = 0;
		size_t loop_c = 0, read_c = 0;
	)
	while (ch != 27 && ch != 10 && ch != 13)
	{
		TIMES(loop_s = chrono::steady_clock::now();)
		if ((!paused || play_once) && video_input)
		{
			TIMES(read_s = chrono::steady_clock::now();)
			if (args.is_single_input)
			{
				// get next frame
				cap >> frame;
				if (frame.empty())
				{
					// try to jump to start and exit if not possible
					if (!cap.set(CAP_PROP_POS_FRAMES, 0))
					{
						cout << "end of video" << endl;
						break;
					}
					cap >> frame;
				}
				// split frame in front and rear
				frame_front = frame(Rect(0, 0, frame.rows, frame.rows));
				frame_rear = frame(Rect(frame.rows, 0, frame.rows, frame.rows));
			}
			else
			{
				// get next frames
				cap_front >> frame_front;
				cap_rear >> frame_rear;
				if (frame_front.empty() || frame_rear.empty())
				{
					// try to jump to start and exit if not possible
					if (!cap_front.set(CAP_PROP_POS_FRAMES, 0))
					{
						cout << "end of front video" << endl;
						break;
					}
					if (!cap_rear.set(CAP_PROP_POS_FRAMES, 0))
					{
						cout << "end of rear video" << endl;
						break;
					}
					cap_front >> frame_front;
					cap_rear >> frame_rear;
				}
			}
			TIMES(read_e = chrono::steady_clock::now(); read_sum += chrono::duration_cast<std::chrono::duration<double>>(read_e - read_s).count(); read_c++;)
			play_once = false;
		}

		TIMES(slid_s = chrono::steady_clock::now();)
		// read slider values
		// params.x_off_front = slider_params.x_off_front / (double) SLIDER_FACTOR_DEFAULT;
		// params.y_off_front = slider_params.y_off_front / (double) SLIDER_FACTOR_DEFAULT;
		params.x_off_front = slider_params.x_off_front - SLIDER_FACTOR_DEFAULT;
		params.y_off_front = slider_params.y_off_front - SLIDER_FACTOR_DEFAULT;
		params.rot_front = deg2rad((slider_params.rot_front - SLIDER_ROTATION_DEFAULT) / 10.)
							+ deg2rad((slider_params.rot_both - SLIDER_ROTATION_DEFAULT) / 10.);
		params.radius_front = (slider_params.radius_front / (double)SLIDER_FACTOR_DEFAULT * 0.1 + 0.9)
							* M_PI / deg2rad(FOV_DEFAULT + slider_params.fov_front - SLIDER_FOV_DEFAULT);

		// params.x_off_rear = slider_params.x_off_rear / (double) SLIDER_FACTOR_DEFAULT;
		// params.y_off_rear = slider_params.y_off_rear / (double) SLIDER_FACTOR_DEFAULT;
		params.x_off_rear = slider_params.x_off_rear - SLIDER_FACTOR_DEFAULT;
		params.y_off_rear = slider_params.y_off_rear - SLIDER_FACTOR_DEFAULT;
		params.rot_rear = deg2rad((slider_params.rot_rear - SLIDER_ROTATION_DEFAULT) / 10.)
						- deg2rad((slider_params.rot_both - SLIDER_ROTATION_DEFAULT) / 10.);
		// params.radius_rear = (slider_params.radius_rear / (double)SLIDER_FACTOR_DEFAULT * 0.1 + 0.9)
		// 					* (M_PI / deg2rad(FOV_DEFAULT + slider_params.fov_rear - SLIDER_FOV_DEFAULT));
		params.radius_rear = (slider_params.radius_rear / (double)SLIDER_FACTOR_DEFAULT * 0.1 + 0.9)
							* (M_PI / deg2rad(FOV_DEFAULT + slider_params.fov_front - SLIDER_FOV_DEFAULT));

		params.orientation = deg2rad(slider_params.orientation);
		params.blend_size = deg2rad(slider_params.blend_size);
		params.blend_limit = M_PI_2 + params.blend_size;
		params.front_limit = M_PI_2 - params.blend_size;
		params.blend_size *= 2;
		TIMES(slid_e = chrono::steady_clock::now(); slid_sum += chrono::duration_cast<std::chrono::duration<double>>(slid_e - slid_s).count();)

		TIMES(prev_s = chrono::steady_clock::now();)
		// clear circles
		drawframe_front = frame_front.clone();
		drawframe_rear = frame_rear.clone();

		// Center coordinates for preview
		center_f.x = height_2 + params.x_off_front;
		center_f.y = height_2 + params.y_off_front;
		center_r.x = height_2 + params.x_off_rear;
		center_r.y = height_2 + params.y_off_rear;

		radius_f = height_2 * params.radius_front;
		radius_r = height_2 * params.radius_rear;
		radius_blend_f = radius_f * (2 - params.front_limit / M_PI_2);
		radius_blend_r = radius_r * params.blend_limit / M_PI_2;
		line_f.x = center_f.x + radius_f * cos(params.rot_front - M_PI_2);
		line_f.y = center_f.y + radius_f * sin(params.rot_front - M_PI_2);
		line_r.x = center_r.x + radius_r * cos(params.rot_rear - M_PI_2);
		line_r.y = center_r.y + radius_r * sin(params.rot_rear - M_PI_2);

		// draw circles and lines
		circle(drawframe_front, center_f, radius_blend_f, prev_blend_color, line_width);
		circle(drawframe_rear, center_r, radius_blend_r, prev_blend_color, line_width);
		circle(drawframe_front, center_f, radius_f, prev_color, line_width);
		circle(drawframe_rear, center_r, radius_r, prev_color, line_width);
		line(drawframe_front, center_f, line_f, prev_color, line_width);
		line(drawframe_rear, center_r, line_r, prev_color, line_width);
		imshow(CAMERA_FRONT_WINDOW_NAME, drawframe_front);
		imshow(CAMERA_REAR_WINDOW_NAME, drawframe_rear);
		TIMES(prev_e = chrono::steady_clock::now(); prev_sum += chrono::duration_cast<std::chrono::duration<double>>(prev_e - prev_s).count();)

		// mapping
		TIMES(genm_s = chrono::steady_clock::now();)
		gen_equi_mapping_table(width, height, height, cv::INTER_NEAREST, args.is_single_input, true, &params, mapping_table);
		#ifdef WITH_OPENCL
		CL_RETERR_CHECK( extra_data.cmdq.enqueueWriteBuffer(extra_data.map, CL_TRUE, 0, mapping_table.dataend - mapping_table.datastart, mapping_table.data) );
		#endif
		TIMES(genm_e = chrono::steady_clock::now(); genm_sum += chrono::duration_cast<std::chrono::duration<double>>(genm_e - genm_s).count();)
		TIMES(rmap_s = chrono::steady_clock::now();)
		if (args.is_single_input)
			remap(frame, mapping_table, cv::INTER_NEAREST, equiframe, extra_data);
		else
			remap(frame_front, frame_rear, mapping_table, cv::INTER_NEAREST, equiframe, extra_data);
		TIMES(rmap_e = chrono::steady_clock::now(); rmap_sum += chrono::duration_cast<std::chrono::duration<double>>(rmap_e - rmap_s).count();)

		// Auf den Schirm Spoki
		imshow(PREVIEW_WINDOW_NAME, equiframe);

		ch = waitKey(1);
		switch (ch)
		{
		case ' ':
			paused ^= 0x01;
			break;
		case 'k':
			paused ^= 0x01;
			break;
		case '.':
			play_once = true;
			break;
		case ',':
			if (args.is_single_input)
				cap.set(cv::CAP_PROP_POS_FRAMES, cap.get(cv::CAP_PROP_POS_FRAMES) - 2);
			else
			{
				cap_front.set(cv::CAP_PROP_POS_FRAMES, cap_front.get(cv::CAP_PROP_POS_FRAMES) - 2);
				cap_rear.set(cv::CAP_PROP_POS_FRAMES, cap_rear.get(cv::CAP_PROP_POS_FRAMES) - 2);
			}
			play_once = true;
			break;
		case 'l':
			if (args.is_single_input)
				cap.set(cv::CAP_PROP_POS_MSEC, cap.get(cv::CAP_PROP_POS_MSEC) + 10000);
			else
			{
				cap_front.set(cv::CAP_PROP_POS_MSEC, cap_front.get(cv::CAP_PROP_POS_MSEC) + 10000);
				cap_rear.set(cv::CAP_PROP_POS_MSEC, cap_rear.get(cv::CAP_PROP_POS_MSEC) + 10000);
			}
			play_once = true;
			break;
		case 'j':
			if (args.is_single_input)
				cap.set(cv::CAP_PROP_POS_MSEC, cap.get(cv::CAP_PROP_POS_MSEC) - 10000);
			else
			{
				cap_front.set(cv::CAP_PROP_POS_MSEC, cap_front.get(cv::CAP_PROP_POS_MSEC) - 10000);
				cap_rear.set(cv::CAP_PROP_POS_MSEC, cap_rear.get(cv::CAP_PROP_POS_MSEC) - 10000);
			}
			play_once = true;
			break;
		default:
			break;
		}
		TIMES(
			loop_e = chrono::steady_clock::now();
			loop_sum += chrono::duration_cast<std::chrono::duration<double>>(loop_e - loop_s).count();
			loop_c++;
			if (loop_c % 30 == 0)
			{
				cout << fixed << setprecision(4)
					<< "loop number: " << loop_c << '\n'
					<< "  read:    " << setw(9) << read_sum / read_c * 1000 << "ms\n"
					<< "  gen map: " << setw(9) << genm_sum / loop_c * 1000 << "ms\n"
					<< "  remap:   " << setw(9) << rmap_sum / loop_c * 1000 << "ms\n"
					<< "  slider:  " << setw(9) << slid_sum / loop_c * 1000 << "ms\n"
					<< "  preview: " << setw(9) << prev_sum / loop_c * 1000 << "ms\n"
					<< "  loop:    " << setw(9) << loop_sum / loop_c * 1000 << "ms  " << setw(9) << loop_c / loop_sum << "hz\n";
			}
		)
	}

	// print command to run again with current parameters
	cout << argv[0];
	if (args.is_single_input)
		cout << ' ' << args.file;
	else
		cout << ' ' << args.file_front << ' ' << args.file_rear;
	if (args.has_resolution)
		cout << " -r " << width << ' ' << height;
	if (args.has_fps)
		cout << " -f " << args.fps;
	if (args.has_fourcc)
		cout << " -c " << args.fourcc_str;
	if (args.format == output_format::FULL)
		cout << " -m";
	if (args.format == output_format::INTEGER)
		cout << " -i";
	else if (args.format == output_format::OPENCV)
		cout << " -cv";
	if (args.has_output)
		cout << " -o " << args.output_file;
	cout << " -p " << slider_params.x_off_front << ' ' << slider_params.y_off_front << ' ' << slider_params.rot_front << ' ' << slider_params.radius_front << ' ' << slider_params.fov_front
		 << ' ' << slider_params.x_off_rear << ' ' << slider_params.y_off_rear << ' ' << slider_params.rot_rear << ' ' << slider_params.radius_rear << ' ' << slider_params.fov_rear << ' ' << slider_params.blend_size << endl;

	// write output to file if enter was pressed
	if (ch == 10 || ch == 13)
	{
		ofstream output_file(args.output_file);
		output_file << fixed << setprecision(10);
		if (args.format == PARAMETER)
		{
			output_file << params.x_off_front << ' ' << params.y_off_front << ' ' << params.radius_front << ' ' << params.rot_front << ' '
						<< params.x_off_rear << ' ' << params.y_off_rear << ' ' << params.radius_rear << ' ' << params.rot_rear << ' '
						<< params.front_limit << ' ' << params.blend_limit << ' ' << params.blend_size << ' ' << params.orientation;
		}
		else
		{
			mte_t mte;
			gen_equi_mapping_table(width, height, height, cv::INTER_NEAREST, args.is_single_input, false, &params, mapping_table);
			output_file << width << ' ' << height << '\n';
			for (j = 0; j < height; j++)
			{
				for (i = 0; i < width; i++)
				{
					mte = mapping_table.at<mte_t>(j, i);
					switch (args.format)
					{
					case output_format::FULL:
						output_file << mte[0] << ' ' << mte[1] << ' ' << mte[2] << ' ' << mte[3] << ' ' << mte[4] << '\n';
						break;
					case output_format::OPENCV:
						if (mte[4] > 0.5)
							output_file << mte[0] << ' ' << mte[1] << '\n';
						else
							output_file << mte[2] + height << ' ' << mte[3] << '\n';
						break;
					case output_format::INTEGER:
						if (mte[4] > 0.5)
							output_file << (int) round(mte[0]) << ' ' << (int) round(mte[1]) << '\n';
						else
							output_file << (int) round(mte[2] + height) << ' ' << (int) round(mte[3]) << '\n';
						break;
					default:
						break;
					}
				}
			}
		}
		output_file.close();
	}

	// When everything done, release the video capture object
	cap.release();
	// Closes all the frames
	destroyAllWindows();

	return EXIT_SUCCESS;
}
