#include "iostream"
#include "cmath"
#include "fstream"
#include "string"

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
#ifdef PRINT_TIMES
#include "chrono"
#define TIMES(E...) E
#else
#define TIMES(E...)
#endif

using namespace std;
using namespace cv;

typedef Vec<double, 5> Vec5d;

const int FOV_DEFAULT = 190; // >= 180

const int SLIDER_FACTOR_RANGE = 2000, SLIDER_FACTOR_DEFAULT = 1000;
const int SLIDER_ROT_RANGE = 3600, SLIDER_ROT_DEFAULT = 1800;
const int SLIDER_FOV_RANGE = 100, SLIDER_FOV_DEFAULT = 10;
const int SLIDER_BLEND_RANGE = SLIDER_FOV_RANGE / 2, SLIDER_BLEND_DEFAULT = 0;
const string SLIDER_FACTOR_NAME = string(" *= slider / ") + to_string(SLIDER_FACTOR_DEFAULT);
const string SLIDER_X_OFF_NAME = string("x") + SLIDER_FACTOR_NAME;
const string SLIDER_Y_OFF_NAME = string("y") + SLIDER_FACTOR_NAME;
const string SLIDER_RADIUS_NAME = string("radius") + SLIDER_FACTOR_NAME;
const string SLIDER_ROT_NAME = string("rot = ((slider - ") + to_string(SLIDER_ROT_DEFAULT) + ") / 10.0)°";
const string SLIDER_FOV_NAME = string("fov = (") + to_string(FOV_DEFAULT) + string(" + slider - ") + to_string(SLIDER_FOV_DEFAULT) + ")°";
const string SLIDER_BLEND_NAME = string("blend size = (slider * 2)°");

const string DEFAULT_OUTPUT_FILE = "mapping-table.txt";

const char CAMERA_FRONT_WINDOW_NAME[] = "Camera front";
const char CAMERA_REAR_WINDOW_NAME[] = "Camera rear";
const char PREVIEW_WINDOW_NAME[] = "Preview";
const char HELP_DIALOG[] = " (<file> | <file_front> <file_rear>) [--resolution -r <width> <height>] [--fps -f <fps>] [--codec -c <fourcc>]"
						   " [--parameters -p <x_offset_front> <y_offset_front> <rotation_front> <fov_front>"
						   " <radius_front> <x_offset_rear> <y_offset_rear> <rotation_rear> <radius_rear> <fov_rear>] [--opencv -cv | --integer -i]\n\n"
						   "Options:\n"
						   "  --output      -o   mapping table output file\n"
						   "  --parameters  -p   set the inital slider parameters\n"
						   "  --opencv      -cv  output in OpenCV remap format double[x, y]\n"
						   "  --integer     -i   output in integer format int[x, y]\n"
						   " For usage with cameras:\n"
						   "  --resolution  -r   set the camera resolution\n"
						   "  --fps         -f   set the camera framerate\n"
						   "  --codec       -c   set the camera codec in fourcc format\n";

enum mapping_table_format
{
	FULL, // double[x1, y1, x2, y2, blend factor]
	OPENCV, // double[x, y]
	INTEGER // int[x, y]
};

/**
 * @brief Holds calibration parameters
 */
struct calib_params
{
	double x_off_front, y_off_front, rot_front, radius_front, fov_front,
		x_off_rear, y_off_rear, rot_rear, radius_rear, fov_rear, blend_size, blend_limit, front_limit;
};

/**
 * @brief Holds slider parameters
 */
struct calib_slider_params
{
	int x_off_front, y_off_front, rot_front, radius_front, fov_front,
		x_off_rear, y_off_rear, rot_rear, radius_rear, fov_rear, blend_size;
};

/**
 * @brief Hold the arguments parsed from commandline
 */
struct program_args
{
	char *file, *file_front, *file_rear, *fourcc_str;
	string output_file = DEFAULT_OUTPUT_FILE;
	int width, height, fps, fourcc;
	calib_slider_params *params;
	bool is_single_input, has_resolution, has_fps, has_fourcc, has_parameters;
	mapping_table_format format;
};

/**
 * @brief Converts degrees to radians.
 * @param deg degrees
 * @return radians
 */
double inline deg2rad(double deg) { return deg * M_PI / 180.; }

/**
 * @brief Checks if x, y are in range of height and limits them if not.
 * @param x 
 * @param y 
 * @param height 
 */
void clip_borders(double &x, double &y, unsigned int height)
{
	if (x < 0)
		x = 0;
	else if (x >= height)
		x = height - 1;
	if (y < 0)
		y = 0;
	else if (y >= height)
		y = height - 1;
}

void dualfisheye2equirectangular(const Mat &front_img, const Mat &rear_img, Mat &equi_img, int width, int height, const calib_params *const params, Mat &mapping_table)
{
	int i, j;			  // equirectangular index
	double phi, theta;	  // equirectangular polar and azimuthal angle
	double p_x, p_y, p_z; // 3D normalised cartesian fisheye vector
	// polar normalised fisheye coordinates
	// a = angle from +x or -x axis
	// r = radius on fish eye to pixel
	// t = angle from +y or -y on fish eye to pixel
	double a, r, t;
	double bf;	 // blend factor
	double x, y; // fisheye index
	Vec5d mte; // mapping table entry

#pragma omp parallel for private(i, j, phi, theta, p_x, p_y, p_z, a, r, t, bf, x, y, mte)
	for (j = 0; j < height; j++)
	{
		theta = (1.0 - ((double)j / (double)height)) * M_PI;
		for (i = 0; i < width; i++)
		{
			phi = (double)i / (double)width * 2.0 * M_PI;

			p_x = cos(phi) * sin(theta);
			p_y = sin(phi) * sin(theta);
			p_z = cos(theta);

			a = atan2(sqrt(p_y * p_y + p_z * p_z), p_x);

			if (a < params->front_limit) // front
			{
				r = 2.0 * a / params->fov_front * params->radius_front;
				t = atan2(p_z, p_y) + params->rot_front;

				x = ((r * cos(t)) + 1) / 2 * height * params->x_off_front;
				y = ((r * sin(t)) + 1) / 2 * height * params->y_off_front;
				clip_borders(x, y, height);
				
				mapping_table.at<Vec5d>(j, i) = Vec5d(x, y, 0, 0, 1);

				equi_img.at<Vec3b>(j, i) = front_img.at<Vec3b>(round(y), round(x));
			}
			else if (a < params->blend_limit) // blend
			{
				// front
				r = 2 * a / params->fov_front * params->radius_front;
				t = atan2(p_z, p_y) + params->rot_front;

				x = (r * cos(t) + 1) / 2 * height * params->x_off_front;
				y = (r * sin(t) + 1) / 2 * height * params->y_off_front;
				clip_borders(x, y, height);

				bf = 1 - ((a - params->front_limit) / (params->blend_size * 2));

				mte = Vec5d(x, y, 0, 0, bf);

				// rear
				r = 2 * (M_PI - a) / params->fov_rear * params->radius_rear;
				t = atan2(p_z, -p_y) + params->rot_rear;

				x = (r * cos(t) + 1) / 2 * height * params->x_off_rear;
				y = (r * sin(t) + 1) / 2 * height * params->y_off_rear;
				clip_borders(x, y, height);

				mte[2] = x;
				mte[3] = y;
				mapping_table.at<Vec5d>(j, i) = mte;

				equi_img.at<Vec3b>(j, i) = bf * front_img.at<Vec3b>(round(mte[1]), round(mte[0])) + (1 - bf) * rear_img.at<Vec3b>(round(y), round(x));
			}
			else // rear
			{
				r = 2.0 * (M_PI - a) / params->fov_rear * params->radius_rear;
				t = atan2(p_z, -p_y) + params->rot_rear;

				x = (r * cos(t) + 1) / 2 * height * params->x_off_rear;
				y = (r * sin(t) + 1) / 2 * height * params->y_off_rear;
				clip_borders(x, y, height);
				
				mapping_table.at<Vec5d>(j, i) = Vec5d(0, 0, x, y, 0);

				equi_img.at<Vec3b>(j, i) = rear_img.at<Vec3b>(round(y), round(x));
			}
		}
	}
}

//TODO
// void optimize(const Mat &front_img, const Mat &rear_img, Mat &equi_img, int width, int height, calib_params *params, const bool fp_out)
// {
// 	Mat equi_grey_img;
// 	cvtColor(equi_img, equi_grey_img, CV_BGR2GRAY);
//
// 	// name pattern: lbl = left border left
// 	Mat lbl = equi_grey_img(Rect(width / 8, 0, width / 8, height));
// 	Mat lbr = equi_grey_img(Rect(width / 4, 0, width / 8, height));
// 	Mat rbl = equi_grey_img(Rect(width / 2 + width / 8, 0, width / 8, height));
// 	Mat rbr = equi_grey_img(Rect(width / 2 + width / 4, 0, width / 8, height));
// 	Mat lbl_des, lbr_des, rbl_des, rbr_des;
// 	vector<KeyPoint> lbl_kp, lbr_kp, rbl_kp, rbr_kp;
// 	vector<DMatch> lb_dm, rb_dm;
// 	Mat lbl_pre, lbr_pre, rbl_pre, rbr_pre;
// 	Mat lb_pre, rb_pre;
//
// 	Mat marker_front = front_img.clone(), marker_rear = rear_img.clone();
// 	Mat feat_front_img = front_img.clone(), feat_rear_img = rear_img.clone();
// 	Mat desc_front, desc_rear;
// 	// cvtColor(front_img, feat_front_img, CV_BGR2GRAY);
// 	// cvtColor(rear_img, feat_rear_img, CV_BGR2GRAY);
// 	std::vector<KeyPoint> kp_front, kp_rear;
// 	int fast_thresh = 0;
// 	createTrackbar("FAST Threshold", PREVIEW_WINDOW_NAME, &fast_thresh, 255);
// 	Ptr<ORB> orb = ORB::create(100);
// 	Ptr<BFMatcher> bfm = BFMatcher::create();
// 	std::vector<DMatch> dm;
//
// 	namedWindow("lb", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
// 	namedWindow("rb", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
// 	cvResizeWindow("lb", 1920 / 4, 1000);
// 	cvResizeWindow("rb", 1920 / 4, 1000);
//
// 	while (true)
// 	{
// 		// FAST(feat_front_img, kp_front, fast_thresh);
// 		// FAST(feat_rear_img, kp_rear, fast_thresh);
// 		// orb->detectAndCompute(feat_front_img, noArray(), kp_front, desc_front);
// 		// orb->detectAndCompute(feat_rear_img, noArray(), kp_rear, desc_rear);
// 		// drawKeypoints(front_img, kp_front, marker_front, Scalar(0, 0, 255));
// 		// drawKeypoints(rear_img, kp_rear, marker_rear, Scalar(0, 0, 255));
// 		// imshow("FAST marker front", marker_front);
// 		// imshow("FAST marker rear", marker_rear);
// 		// bfm->match(desc_front, desc_rear, dm);
// 		// drawMatches(front_img, kp_front, rear_img, kp_rear, dm, equi_img, Scalar(0, 0, 255), Scalar(255, 0, 0));
// 		// imshow("Matches", equi_img);
//
// 		orb->detectAndCompute(lbl, noArray(), lbl_kp, lbl_des);
// 		orb->detectAndCompute(lbr, noArray(), lbr_kp, lbr_des);
// 		orb->detectAndCompute(rbl, noArray(), rbl_kp, rbl_des);
// 		orb->detectAndCompute(rbr, noArray(), rbr_kp, rbr_des);
// 		drawKeypoints(lbl, lbl_kp, lbl_pre, Scalar(0, 0, 255));
// 		drawKeypoints(lbr, lbr_kp, lbr_pre, Scalar(0, 0, 255));
// 		drawKeypoints(rbl, rbl_kp, rbl_pre, Scalar(0, 0, 255));
// 		drawKeypoints(rbr, rbr_kp, rbr_pre, Scalar(0, 0, 255));
// 		imshow("lbl", lbl_pre);
// 		imshow("lbr", lbr_pre);
// 		imshow("rbl", rbl_pre);
// 		imshow("rbr", rbr_pre);
//
// 		bfm->match(lbl_des, lbr_des, lb_dm);
// 		bfm->match(rbl_des, rbr_des, rb_dm);
// 		drawMatches(lbl, lbl_kp, lbr, lbr_kp, lb_dm, lb_pre, Scalar(0, 0, 255), Scalar(255, 0, 0));
// 		drawMatches(rbl, rbl_kp, rbr, rbr_kp, rb_dm, rb_pre, Scalar(0, 0, 255), Scalar(255, 0, 0));
// 		imshow("lb", lb_pre);
// 		imshow("rb", rb_pre);
//
// 		waitKey(10);
// 	}
// }

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
		else if (strcmp(argv[i], "-cv") == 0 || strcmp(argv[i], "--opencv") == 0)
		{
			if (args->format != mapping_table_format::FULL)
			{
				cout << "Only one output format can be sat at a time." << endl;
				exit(1);
			}
			args->format = mapping_table_format::OPENCV;
		}
		else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--integer") == 0)
		{
			if (args->format != mapping_table_format::FULL)
			{
				cout << "Only one output format can be sat at a time." << endl;
				exit(1);
			}
			args->format = mapping_table_format::INTEGER;
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
}

int main(int argc, char **argv)
{
	int i, j;
	int height, width;
	program_args args = {0};
	calib_params params = {0};
	calib_slider_params slider_params = {SLIDER_FACTOR_DEFAULT, SLIDER_FACTOR_DEFAULT, SLIDER_ROT_DEFAULT, SLIDER_FACTOR_DEFAULT, SLIDER_FOV_DEFAULT,
										 SLIDER_FACTOR_DEFAULT, SLIDER_FACTOR_DEFAULT, SLIDER_ROT_DEFAULT, SLIDER_FACTOR_DEFAULT, SLIDER_FOV_DEFAULT};

	args.params = &slider_params;

	// parse arguments and set optional parameters
	parse_args(argc, argv, &args);

	VideoCapture cap, cap_front, cap_rear;
	if (args.is_single_input)
	{
		// TODO sets the caps but cameras don't change
		// set optional fps for cameras
		if (args.has_fps)
		{
			if (!cap.set(CAP_PROP_FPS, args.fps))
				printf("Couldn't set FPS to %d\n", args.fps);
		}
		// set optional codec for cameras
		if (args.has_fourcc)
		{
			if (!cap.set(CAP_PROP_FOURCC, static_cast<double>(args.fourcc)) && cap.get(CAP_PROP_FOURCC) == args.fourcc)
				printf("Couldn't set codec to %s\n", args.fourcc_str);
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

		// try to open video
		cap.open(args.file);
		if (!cap.isOpened())
		{
			printf("Could not open video %s\n", args.file);
			return EXIT_FAILURE;
		}
		width = cap.get(CAP_PROP_FRAME_WIDTH);
		height = cap.get(CAP_PROP_FRAME_HEIGHT);
	}
	else
	{
		// TODO sets the caps but cameras don't change
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
	createTrackbar(SLIDER_ROT_NAME, CAMERA_FRONT_WINDOW_NAME, &(slider_params.rot_front), SLIDER_ROT_RANGE);
	createTrackbar(SLIDER_FOV_NAME, CAMERA_FRONT_WINDOW_NAME, &(slider_params.fov_front), SLIDER_FOV_RANGE);

	createTrackbar(SLIDER_X_OFF_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.x_off_rear), SLIDER_FACTOR_RANGE);
	createTrackbar(SLIDER_Y_OFF_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.y_off_rear), SLIDER_FACTOR_RANGE);
	createTrackbar(SLIDER_RADIUS_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.radius_rear), SLIDER_FACTOR_RANGE);
	createTrackbar(SLIDER_ROT_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.rot_rear), SLIDER_ROT_RANGE);
	createTrackbar(SLIDER_FOV_NAME, CAMERA_REAR_WINDOW_NAME, &(slider_params.fov_rear), SLIDER_FOV_RANGE);

	createTrackbar(SLIDER_BLEND_NAME, PREVIEW_WINDOW_NAME, &(slider_params.blend_size), SLIDER_BLEND_RANGE);

	//TODO optimize button
	// int optimization = 0;
	// createTrackbar("optimze", PREVIEW_WINDOW_NAME, &optimization, 1);

	Mat frame, frame_front, frame_rear, equiframe, drawframe_front, drawframe_rear;
	Point center_f, center_r, line_f, line_r;
	int radius_f, radius_r, ch = 0, height_2 = height / 2, line_width = width / 1000 + 1;
	bool pause = false, video_input = true;

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
	// create mapping table
	Mat mapping_table(height, width, CV_64FC(5));

	printf("Controls:\n\tspace:\tstart/stop videoplayback\n\t<-/->:\tdecrement/increment current slider\n\ttab:\tnext slider\n"
		   "\tenter:\tsave to file and exit\n\tesc:\texit without saving\n");

	TIMES(chrono::time_point<chrono::steady_clock> start, end);
	TIMES(chrono::duration<int64_t, std::nano> loop_time);
	while (ch != 27 && ch != 10 && ch != 13)
	{
		TIMES(start = chrono::steady_clock::now());
		if (!pause && video_input)
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
		}
		// clear circles
		drawframe_front = frame_front.clone();
		drawframe_rear = frame_rear.clone();

		// read slider values
		params.x_off_front = slider_params.x_off_front / (double) SLIDER_FACTOR_DEFAULT;
		params.y_off_front = slider_params.y_off_front / (double) SLIDER_FACTOR_DEFAULT;
		params.rot_front = deg2rad((slider_params.rot_front - SLIDER_ROT_DEFAULT) / 10.);
		params.radius_front = slider_params.radius_front / (double)SLIDER_FACTOR_DEFAULT;
		params.fov_front = deg2rad(FOV_DEFAULT + slider_params.fov_front - SLIDER_FOV_DEFAULT);

		params.x_off_rear = slider_params.x_off_rear / (double) SLIDER_FACTOR_DEFAULT;
		params.y_off_rear = slider_params.y_off_rear / (double) SLIDER_FACTOR_DEFAULT;
		params.rot_rear = deg2rad((slider_params.rot_rear - SLIDER_ROT_DEFAULT) / 10.);
		params.radius_rear = slider_params.radius_rear / (double)SLIDER_FACTOR_DEFAULT;
		params.fov_rear = deg2rad(FOV_DEFAULT + slider_params.fov_rear - SLIDER_FOV_DEFAULT);

		params.blend_size = deg2rad(slider_params.blend_size);
		params.blend_limit = M_PI_2 + params.blend_size;
		params.front_limit = M_PI_2 - params.blend_size;

		// Center coordinates for preview
		center_f.x = height_2 * params.x_off_front;
		center_f.y = height_2 * params.y_off_front;
		center_r.x = height_2 * params.x_off_rear;
		center_r.y = height_2 * params.y_off_rear;

		radius_f = height_2 * params.radius_front;
		radius_r = height_2 * params.radius_rear;
		line_f.x = center_f.x + radius_f * cos(params.rot_front - M_PI_2);
		line_f.y = center_f.y + radius_f * sin(params.rot_front - M_PI_2);
		line_r.x = center_r.x + radius_r * cos(params.rot_rear - M_PI_2);
		line_r.y = center_r.y + radius_r * sin(params.rot_rear - M_PI_2);

		// draw circles and lines
		circle(drawframe_front, center_f, radius_f, Scalar(0, 255, 0), line_width, 0, 0);
		circle(drawframe_rear, center_r, radius_r, Scalar(0, 0, 255), line_width, 0, 0);
		line(drawframe_front, center_f, line_f, Scalar(0, 255, 0), line_width);
		line(drawframe_rear, center_r, line_r, Scalar(0, 0, 255), line_width);
		imshow(CAMERA_FRONT_WINDOW_NAME, drawframe_front);
		imshow(CAMERA_REAR_WINDOW_NAME, drawframe_rear);

		// mapping
		dualfisheye2equirectangular(frame_front, frame_rear, equiframe, width, height, &params, mapping_table);
		// Auf den Schirm Spoki
		imshow(PREVIEW_WINDOW_NAME, equiframe);

		TIMES(end = chrono::steady_clock::now());
		TIMES(loop_time = end - start);
		TIMES(cout << "frametime: " << chrono::duration_cast<chrono::milliseconds>(loop_time).count() << "ms" << endl);

		//TODO optimize
		// if (optimization)
		// 	optimize(frame_front, frame_rear, equiframe, width, height, &params);

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
	cout << " -p " << slider_params.x_off_front << ' ' << slider_params.y_off_front << ' ' << slider_params.rot_front << ' ' << slider_params.radius_front << ' ' << slider_params.fov_front
		 << ' ' << slider_params.x_off_rear << ' ' << slider_params.y_off_rear << ' ' << slider_params.rot_rear << ' ' << slider_params.radius_rear << ' ' << slider_params.fov_rear << ' ' << slider_params.blend_size << endl;

	// write mappings to file if enter was pressed
	if (ch == 10 || ch == 13)
	{
		Vec5d mte;
		ofstream mapping_file(args.output_file);
		mapping_file << width << ' ' << height << '\n';
		for (j = 0; j < height; j++)
		{
			for (i = 0; i < width; i++)
			{
				mte = mapping_table.at<Vec5d>(j, i);
				switch (args.format)
				{
				case mapping_table_format::FULL:
					// mapping_file << mte << '\n';
					mapping_file << mte[0] << ' ' << mte[1] << ' ' << mte[2] << ' ' << mte[3] << ' ' << mte[4] << '\n';
					break;
				case mapping_table_format::OPENCV:
					if (mte[4] > 0.5)
						mapping_file << mte[0] << ' ' << mte[1] << '\n';
					else
						mapping_file << mte[2] + height << ' ' << mte[3] << '\n';
					break;
				case mapping_table_format::INTEGER:
					if (mte[4] > 0.5)
						mapping_file << (int) round(mte[0]) << ' ' << (int) round(mte[1]) << '\n';
					else
						mapping_file << (int) round(mte[2] + height) << ' ' << (int) round(mte[3]) << '\n';
					break;
				default:
					break;
				}
			}
		}
		mapping_file.close();
	}

	// When everything done, release the video capture object
	cap.release();
	// Closes all the frames
	destroyAllWindows();

	return EXIT_SUCCESS;
}
