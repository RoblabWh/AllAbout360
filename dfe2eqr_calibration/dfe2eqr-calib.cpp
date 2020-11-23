#include "iostream"
#include "cmath"
#include "fstream"
#include "string"
#include "chrono"
#include <time.h>

#include <opencv2/opencv.hpp>
#ifdef _OPENMP
#include "omp.h"
#endif

// g++ -O2 -fopenmp -o dfe2eqr-calib dfe2eqr-calib.cpp -lm `pkg-config --cflags opencv` `pkg-config --libs opencv`

#ifdef DEBUG
#define DBG(E...) E
#else
#define DBG(E...)
#endif

#define RAD_TO_DEG(R) ((R)*180.0 / M_PI)
#define DEG_TO_RAD(D) ((D)*M_PI / 180.0)

using namespace std;
using namespace cv;

const int FOV_DEFAULT = 190; // >= 180

const int SLIDER_PX_OFF_RANGE = 200, SLIDER_PX_OFF_DEFAULT = 100;
const int SLIDER_ROT_RANGE = 200, SLIDER_ROT_DEFAULT = 100;
const int SLIDER_FOV_RANGE = 90, SLIDER_FOV_DEFAULT = FOV_DEFAULT - 180;
const string SLIDER_PX_OFF_NAME = string(" += slider - ") + to_string(SLIDER_PX_OFF_DEFAULT);
const string SLIDER_X_OFF_NAME = string("x") + SLIDER_PX_OFF_NAME;
const string SLIDER_Y_OFF_NAME = string("y") + SLIDER_PX_OFF_NAME;
const string SLIDER_ROT_NAME = string("rot = ((slider - ") + to_string(SLIDER_ROT_DEFAULT) + ") / 10.0)°";
const string SLIDER_RADIUS_NAME = string("radius") + SLIDER_PX_OFF_NAME;
const string SLIDER_FOV_NAME = string("fov = (") + to_string(FOV_DEFAULT) + string(" + slider - ") + to_string(SLIDER_FOV_DEFAULT) + ")°";

const char CAMERA_FRONT_WINDOW_NAME[] = "Camera front";
const char CAMERA_REAR_WINDOW_NAME[] = "Camera rear";
const char PREVIEW_WINDOW_NAME[] = "Preview";
const char HELP_DIALOG[] = " (<file> | <file_front> <file_rear>) [--resolution -r <width> <height>] [--fps -f <fps>] [--codec -c <fourcc>]"
						   " [--parameters -p <x_offset_front> <y_offset_front> <rotation_front> <fov_front>"
						   " <radius_front> <x_offset_rear> <y_offset_rear> <rotation_rear> <radius_rear> <fov_rear>]";

Mat *mapx, *mapy;

struct calib_params
{
	int x_off_front, y_off_front, rot_front, radius_front, fov_front,
		x_off_rear, y_off_rear, rot_rear, radius_rear, fov_rear;
};
struct program_args
{
	char *file, *file_front, *file_rear, *fourcc_str;
	int width, height, fps, fourcc;
	calib_params params;
	bool is_single_input, has_resolution, has_fps, has_fourcc, has_parameters;
};

Vec3b get_px_pos(const Mat &front_img, const Mat &rear_img, int half_width, float r, float t, bool is_front, int i, int j, const calib_params *const params)
{
	float x, y;

	x = ((r * cos(t)) + 1) / 2; // normalised betwen 0 and 1
	y = ((r * sin(t)) + 1) / 2; // normalised betwen 0 and 1

	if (is_front)
	{
		x *= (half_width + params->radius_front); // radius_offset
		y *= (half_width + params->radius_front);
		x += params->x_off_front; // xf_offset
		y += params->y_off_front; // yf_offset
	}
	else
	{
		x *= (half_width + params->radius_rear); // radius_offset
		y *= (half_width + params->radius_rear);
		x += params->x_off_rear; //xr_offset
		y += params->y_off_rear; //yr_offset
	}
	// dont access bad memory, results in image "errors" with bad parameters
	if (x < 0)
		x = 0;
	else if (x >= half_width)
		x = half_width - 1;
	if (y < 0)
		y = 0;
	else if (y >= half_width)
		y = half_width - 1;

	// save mapping
	mapx->at<unsigned short>(j, i) = is_front
										 ? (unsigned short)x
										 : half_width + (unsigned short)x;
	mapy->at<unsigned short>(j, i) = (unsigned short)y;

	return is_front
			   ? front_img.at<Vec3b>((int)y, (int)x)
			   : rear_img.at<Vec3b>((int)y, (int)x);
}

int dualfisheye2equirectangular(const Mat &front_img, const Mat &rear_img, Mat &equi_img, int width, int height, const calib_params *const params)
{
	float frontlimit = DEG_TO_RAD(90.0);
	float blendlimit = DEG_TO_RAD(90.0);
	float blendsize = DEG_TO_RAD(0.0); // not supportend e.g. 16 (8+8)
	float fovf = DEG_TO_RAD(FOV_DEFAULT + params->fov_front);
	float fovr = DEG_TO_RAD(FOV_DEFAULT + params->fov_rear);
	float semi = DEG_TO_RAD(180.0);
	int i, j;
	float phi, theta, a, r, t, x, y, z;
	float bf, ar, rr, tr, bfr;

#pragma omp parallel for private(phi, theta, a, r, t, x, y, z, i, j, bf, ar, rr, tr, bfr)
	for (j = 0; j < height; j++)
	{
		phi = (1.0 - ((float)j / (float)height)) * M_PI;
		for (i = 0; i < width; i++)
		{
			theta = (float)i / (float)width * 2.0 * M_PI;
			//  # 3D normalised cartesian
			x = cos(theta) * sin(phi);
			y = sin(theta) * sin(phi);
			z = cos(phi);
			// normalised fisheye coordinates
			//  # a = angle from +x or -x axis
			//  # r = radius on fish eye to pixel
			//  # t = angle from +y or -y on fish eye to pixel
			a = atan2(sqrt(y * y + z * z), x);
			if (a < frontlimit)
			{ //  # front
				r = 2.0 * a / fovf;
				t = atan2(z, y) - DEG_TO_RAD(params->rot_front / 10.);
				equi_img.at<Vec3b>(j, i) = get_px_pos(front_img, rear_img, height, r, t, true, i, j, params);
			}
			else if (a < blendlimit)
			{ //  # blend
				// fr
				r = 2 * a / fovf;
				t = atan2(z, y) - DEG_TO_RAD(params->rot_front / 10.);
				bf = 1 - ((a - frontlimit) / blendsize);
				// re
				ar = semi - a;
				rr = 2 * ar / fovr;
				tr = atan2(z, -y) - DEG_TO_RAD(params->rot_rear / 10.);
				bfr = 1 - bf;
				//print(size,r,t,rr,tr, bfr);
				if (rr > 0.999)
					rr = 0.999;
				equi_img.at<Vec3b>(j, i) = bf * get_px_pos(front_img, rear_img, height, r, t, true, i, j, params) + bfr * get_px_pos(front_img, rear_img, height, r, t, false, i, j, params);
			}
			else
			{ //  # rear
				r = 2.0 * (semi - a) / fovr;
				t = atan2(z, -y) - DEG_TO_RAD(params->rot_rear / 10.);
				equi_img.at<Vec3b>(j, i) = get_px_pos(front_img, rear_img, height, r, t, false, i, j, params);
			}
		}
	}

	return 0;
}

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
			args->fourcc = CV_FOURCC(argv[i][0], argv[i][1], argv[i][2], argv[i][3]);
			args->fourcc_str = argv[i];
			args->has_fourcc = true;
		}
		else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--parameters") == 0)
		{
			if (i >= argc - 10)
			{
				cout << "All paramters have to be given." << endl;
				exit(1);
			}
			args->params.x_off_front = stoi(argv[i + 1]);
			args->params.y_off_front = stoi(argv[i + 2]);
			args->params.rot_front = stoi(argv[i + 3]);
			args->params.radius_front = stoi(argv[i + 4]);
			args->params.fov_front = stoi(argv[i + 5]);

			args->params.x_off_rear = stoi(argv[i + 6]);
			args->params.y_off_rear = stoi(argv[i + 7]);
			args->params.rot_rear = stoi(argv[i + 8]);
			args->params.radius_rear = stoi(argv[i + 9]);
			args->params.fov_rear = stoi(argv[i + 10]);

			args->has_parameters = true;
			i += 10;
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			cout << argv[0] << HELP_DIALOG << endl;
			exit(0);
		}
		else
		{
			cout << "unknown argument \"" << argv[i] << "\"" << endl;
			exit(1);
		}
	}
}

int main(int argc, char **argv)
{
	int i, j;
	program_args args = {0};
	calib_params *params = &args.params, slider_params = {SLIDER_PX_OFF_DEFAULT, SLIDER_PX_OFF_DEFAULT, SLIDER_ROT_DEFAULT, SLIDER_PX_OFF_DEFAULT, SLIDER_FOV_DEFAULT,
														  SLIDER_PX_OFF_DEFAULT, SLIDER_PX_OFF_DEFAULT, SLIDER_ROT_DEFAULT, SLIDER_PX_OFF_DEFAULT, SLIDER_FOV_DEFAULT};
	int height, width;

	// parse arguments and set optional parameters
	parse_args(argc, argv, &args);
	if (args.has_parameters)
	{
		slider_params.x_off_front += params->x_off_front;
		slider_params.y_off_front += params->y_off_front;
		slider_params.rot_front += params->rot_front;
		slider_params.radius_front += params->radius_front;
		slider_params.fov_front += params->fov_front;

		slider_params.x_off_rear += params->x_off_rear;
		slider_params.y_off_rear += params->y_off_rear;
		slider_params.rot_rear += params->rot_rear;
		slider_params.radius_rear += params->radius_rear;
		slider_params.fov_rear += params->fov_rear;
	}

	VideoCapture cap, cap_front, cap_rear;
	if (args.is_single_input)
	{
		// try to open video
		cap.open(args.file);
		if (!cap.isOpened())
		{
			printf("Could not open video %s\n", args.file);
			return -1;
		}

		// set optional resolution for cameras
		if (args.has_resolution)
		{
			width = args.width;
			height = args.height;
			// try to set given resolution and check success
			if (!cap.set(CAP_PROP_FRAME_WIDTH, width))
				printf("Couldn't set width to %d\n", width);
			if (!cap.set(CAP_PROP_FRAME_HEIGHT, height))
				printf("Couldn't set heigth to %d\n", height);
		}
		else
		{
			width = cap.get(CAP_PROP_FRAME_WIDTH);
			height = cap.get(CAP_PROP_FRAME_HEIGHT);
		}
		// set optional fps for cameras
		if (args.has_fps)
		{
			if (!cap.set(CAP_PROP_FPS, args.fps))
				printf("Couldn't set FPS to %d\n", args.fps);
		}
		// set optional codec for cameras
		if (args.has_fourcc)
		{
			if (!cap.set(CAP_PROP_FOURCC, static_cast<double>(args.fourcc)))
				printf("Couldn't set codec to %s\n", args.fourcc_str);
		}
	}
	else
	{
		cap_front.open(args.file_front);
		cap_rear.open(args.file_rear);
		if (!cap_front.isOpened())
		{
			printf("Could not open front video %s\n", args.file_front);
			return -1;
		}
		if (!cap_rear.isOpened())
		{
			printf("Could not open rear video %s\n", args.file_rear);
			return -1;
		}

		if (args.has_resolution)
		{
			width = args.width;
			height = args.height;
			// try to set given resolution and check success
			if (!cap_front.set(CAP_PROP_FRAME_WIDTH, height))
				printf("Couldn't set front width to %d\n", height);
			if (!cap_front.set(CAP_PROP_FRAME_HEIGHT, height))
				printf("Couldn't set front heigth to %d\n", height);
			if (!cap_rear.set(CAP_PROP_FRAME_WIDTH, height))
				printf("Couldn't set rear width to %d\n", height);
			if (!cap_rear.set(CAP_PROP_FRAME_HEIGHT, height))
				printf("Couldn't set rear heigth to %d\n", height);
		}
		else
		{
			if (cap_front.get(CAP_PROP_FRAME_WIDTH) != cap_rear.get(CAP_PROP_FRAME_WIDTH) || cap_front.get(CAP_PROP_FRAME_HEIGHT) != cap_rear.get(CAP_PROP_FRAME_HEIGHT))
			{
				printf("Resolution of the front and rear video differ.\n");
				return -1;
			}
			width = cap_front.get(CAP_PROP_FRAME_WIDTH) * 2;
			height = cap_front.get(CAP_PROP_FRAME_HEIGHT);
		}
		if (args.has_fps)
		{
			if (!cap_front.set(CAP_PROP_FPS, args.fps))
				printf("Couldn't set front FPS to %d\n", args.fps);
			if (!cap_rear.set(CAP_PROP_FPS, args.fps))
				printf("Couldn't set rear FPS to %d\n", args.fps);
		}
		if (args.has_fourcc)
		{
			if (!cap_front.set(CAP_PROP_FOURCC, static_cast<double>(args.fourcc)))
				printf("Couldn't set front codec to %s\n", args.fourcc_str);
			if (!cap_rear.set(CAP_PROP_FOURCC, static_cast<double>(args.fourcc)))
				printf("Couldn't set rear codec to %s\n", args.fourcc_str);
		}
	}

	// create windows
	namedWindow(CAMERA_FRONT_WINDOW_NAME, WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	namedWindow(CAMERA_REAR_WINDOW_NAME, WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	namedWindow(PREVIEW_WINDOW_NAME, WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	cvResizeWindow(CAMERA_FRONT_WINDOW_NAME, 720, 720);
	cvResizeWindow(CAMERA_REAR_WINDOW_NAME, 720, 720);
	cvResizeWindow(PREVIEW_WINDOW_NAME, 1440, 720);

	// create slider
	createTrackbar(SLIDER_X_OFF_NAME, CAMERA_FRONT_WINDOW_NAME, &(slider_params.x_off_front), SLIDER_PX_OFF_RANGE);
	createTrackbar(SLIDER_Y_OFF_NAME, CAMERA_FRONT_WINDOW_NAME, &(slider_params.y_off_front), SLIDER_PX_OFF_RANGE);
	createTrackbar(SLIDER_ROT_NAME, CAMERA_FRONT_WINDOW_NAME, &(slider_params.rot_front), SLIDER_ROT_RANGE);
	createTrackbar(SLIDER_RADIUS_NAME, CAMERA_FRONT_WINDOW_NAME, &(slider_params.radius_front), SLIDER_PX_OFF_RANGE);
	createTrackbar(SLIDER_FOV_NAME, CAMERA_FRONT_WINDOW_NAME, &(slider_params.fov_front), SLIDER_FOV_RANGE);

	createTrackbar(SLIDER_X_OFF_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.x_off_rear), SLIDER_PX_OFF_RANGE);
	createTrackbar(SLIDER_Y_OFF_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.y_off_rear), SLIDER_PX_OFF_RANGE);
	createTrackbar(SLIDER_ROT_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.rot_rear), SLIDER_ROT_RANGE);
	createTrackbar(SLIDER_RADIUS_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.radius_rear), SLIDER_PX_OFF_RANGE);
	createTrackbar(SLIDER_FOV_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.fov_rear), SLIDER_FOV_RANGE);

	Mat frame, frame_front, frame_rear, equiframe, circleframe_front, circleframe_rear;
	CvPoint center_f, center_r;
	int radius_f, radius_r, ch = 0;
	bool pause = true;

	// get first image and check if the file is empty
	if (args.is_single_input)
	{
		cap >> frame;
		if (frame.empty())
		{
			printf("Video is empty.\n");
			return -1;
		}
				// split frame in front and rear
		frame_front = frame(Rect(0, 0, height, height));
		frame_rear = frame(Rect(height, 0, height, height));
	}
	else
	{
		cap_front >> frame_front;
		if (frame_front.empty())
		{
			printf("Front video is empty.\n");
			return -1;
		}
		cap_rear >> frame_rear;
		if (frame_rear.empty())
		{
			printf("Rear video is empty.\n");
			return -1;
		}
	}
	// init outout image
	equiframe.create(height, width, frame_front.type());
	// create maps
	mapx = new Mat(height, width, CV_16UC1);
	mapy = new Mat(height, width, CV_16UC1);

	printf("Controls:\n\tspace:\tstart/stop videoplayback\n\t<-/->:\tdecrement/increment current slider\n\ttab:\tnext slider\n"
		   "\tenter:\tsave to file and exit\n\tesc:\texit without saving\n");

	chrono::time_point<chrono::steady_clock> start, end;
	chrono::duration<int64_t, std::nano> loop_time;
	while (ch != 27 && ch != 10 && ch != 13)
	{
		start = chrono::steady_clock::now();
		if (!pause)
		{
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
				frame_front = frame(Rect(0, 0, height, height));
				frame_rear = frame(Rect(height, 0, height, height));
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
		}
		// clear circles
		circleframe_front = frame_front.clone();
		circleframe_rear = frame_rear.clone();

		// read slider values
		params->x_off_front = slider_params.x_off_front - SLIDER_PX_OFF_DEFAULT;
		params->y_off_front = slider_params.y_off_front - SLIDER_PX_OFF_DEFAULT;
		params->rot_front = slider_params.rot_front - SLIDER_ROT_DEFAULT;
		params->radius_front = slider_params.radius_front - SLIDER_PX_OFF_DEFAULT;
		params->fov_front = slider_params.fov_front - SLIDER_FOV_DEFAULT;

		params->x_off_rear = slider_params.x_off_rear - SLIDER_PX_OFF_DEFAULT;
		params->y_off_rear = slider_params.y_off_rear - SLIDER_PX_OFF_DEFAULT;
		params->rot_rear = slider_params.rot_rear - SLIDER_ROT_DEFAULT;
		params->radius_rear = slider_params.radius_rear - SLIDER_PX_OFF_DEFAULT;
		params->fov_rear = slider_params.fov_rear - SLIDER_FOV_DEFAULT;

		// Center coordinates for circle preview
		center_f.x = (int)(width / 4 + params->x_off_front);
		center_f.y = (int)(height / 2 + params->y_off_front);
		center_r.x = (int)(width / 4 + params->x_off_rear);
		center_r.y = (int)(height / 2 + params->y_off_rear);
		radius_f = (int)(height / 2 + params->radius_front);
		radius_r = (int)(height / 2 + params->radius_rear);

		// draw circles
		circle(circleframe_front, center_f, radius_f, Scalar(0, 255, 0), 2, 0, 0);
		circle(circleframe_rear, center_r, radius_r, Scalar(0, 0, 255), 2, 0, 0);
		imshow(CAMERA_FRONT_WINDOW_NAME, circleframe_front);
		imshow(CAMERA_REAR_WINDOW_NAME, circleframe_rear);

		// mapping
		dualfisheye2equirectangular(frame_front, frame_rear, equiframe, width, height, params);
		// Auf den Schirm Spoki
		imshow(PREVIEW_WINDOW_NAME, equiframe);

		end = chrono::steady_clock::now();
		loop_time = end - start;
		DBG(cout << "frametime: " << chrono::duration_cast<chrono::milliseconds>(loop_time).count() << "ms" << endl;)

		ch = waitKey(1);
		if (ch == ' ')
			pause ^= 0x01;
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
	cout << " -p " << params->x_off_front << ' ' << params->y_off_front << ' ' << params->rot_front << ' ' << params->radius_front << ' ' << params->fov_front
		 << ' ' << params->x_off_rear << ' ' << params->y_off_rear << ' ' << params->rot_rear << ' ' << params->radius_rear << ' ' << params->fov_rear << endl;

	// write mappings to file if enter was pressed
	if (ch == 10 || ch == 13)
	{
		ofstream mapping_file("mappingfile");
		mapping_file << width << ' ' << height << '\n';
		for (j = 0; j < height; j++)
		{
			for (i = 0; i < width; i++)
			{
				mapping_file << mapx->at<unsigned short>(j, i) << ' ' << mapy->at<unsigned short>(j, i) << '\n';
			}
		}
		mapping_file.close();
	}

	// When everything done, release the video capture object
	cap.release();
	// Closes all the frames
	destroyAllWindows();

	return 0;
}
