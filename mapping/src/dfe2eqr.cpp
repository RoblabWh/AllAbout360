#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif

// g++ -O2 -fopenmp -o dfe2eqr dfe2eqr.cpp  `pkg-config --cflags opencv` `pkg-config --libs opencv`

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

using namespace std;

namespace cv
{
	typedef cv::Vec<double, 5> Vec5d;
}

const string WINDOW_NAME = "Output";

/**
 * @brief Reads the mapping table from file at path
 * 
 * @param path path to mapping file
 * @param mapping_table mapping table in format [x1, y1, x2, y2, blend factor]
 */
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
	cout << width << ' ' << height << endl;
	mapping_table.create(height, width, CV_64FC(5));
	cv::Vec5d mte;
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
				mapping_table.at<cv::Vec5d>(j, i) = mte;
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
 * @param map_file_path path to file with mapping table
 * @param vid_path_1 path to first or only video input (file or device)
 * @param vid_path_2 path to second video input
 */
void parse_args(int argc, char **argv, string &map_file_path, string &vid_path_1, string &vid_path_2, string &out_path)
{
	// check for arguments
	if (argc < 3 || argc > 6)
	{
		printf("usage:\n");
		printf("\t%s <video> <mapping table> [-o <output file>]\n", argv[0]);
		printf("\t%s <video_front> <video_rear> <mapping table> [-o <output file>]\n", argv[0]);
		printf("example:\n");
		printf("\t%s example.mp4 mapping-table.txt\n", argv[0]);
		printf("\t%s example_front.mp4 example_rear.mp4 mapping-table.txt\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	// read arguments
	if (argc == 3)
	{
		vid_path_1 = argv[1];
		map_file_path = argv[2];
		cout << "Processing video file \"" << vid_path_1 << "\" with mapping table \"" << map_file_path << "\"." << endl;
	}
	else if (argc == 4)
	{
		vid_path_1 = argv[1];
		vid_path_2 = argv[2];
		map_file_path = argv[3];
		cout << "Processing video files \"" << vid_path_1 << "\" and \"" << vid_path_2 << "\" with mapping table \"" << map_file_path << "\"." << endl;
	}
	//TODO better error checking and parsing
	else if (argc == 5)
	{
		vid_path_1 = argv[1];
		map_file_path = argv[2];
		cout << "Processing video file \"" << vid_path_1 << "\" with mapping table \"" << map_file_path << "\"." << endl;
		out_path = argv[4];
	}
	else if (argc == 6)
	{
		vid_path_1 = argv[1];
		vid_path_2 = argv[2];
		map_file_path = argv[3];
		cout << "Processing video files \"" << vid_path_1 << "\" and \"" << vid_path_2 << "\" with mapping table \"" << map_file_path << "\"." << endl;
		out_path = argv[5];
	}
}

/**
 * @brief Remaps input to output based on mapping table
 * 
 * @param in input
 * @param map mapping table
 * @param out output
 */
void remap(const cv::Mat &in, const cv::Mat &map, cv::Mat &out)
{
	cv::Vec5d mte;
#pragma omp parallel for collapse(2) schedule(dynamic, 2048) private(mte)
	for (int i = 0; i < map.rows; i++)
	{
		for (int j = 0; j < map.cols; j++)
		{
			mte = map.at<cv::Vec5d>(i, j);
			out.at<cv::Vec3b>(i, j) = in.at<cv::Vec3b>(mte[1], mte[0]) * mte[4] + in.at<cv::Vec3b>(mte[3], mte[2] + map.rows) * (1 - mte[4]);
		}
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
void remap(const cv::Mat &in_1, const cv::Mat &in_2, const cv::Mat &map, cv::Mat &out)
{
	cv::Vec5d mte;
#pragma omp parallel for collapse(2) schedule(dynamic, 2048) private(mte)
	for (int i = 0; i < map.rows; i++)
	{
		for (int j = 0; j < map.cols; j++)
		{
			mte = map.at<cv::Vec5d>(i, j);
			out.at<cv::Vec3b>(i, j) = in_1.at<cv::Vec3b>(mte[1], mte[0]) * mte[4] + in_2.at<cv::Vec3b>(mte[3], mte[2]) * (1 - mte[4]);
		}
	}
}

template <typename T, typename ACC_FN>
void loop(T in, ACC_FN fn, const cv::Mat &map, cv::Mat &equiframe)
{
	cv::Mat frame;
	TIMES(
		unsigned int nrframes = 0; double mapping_sum = 0, loading_sum = 0, preview_sum = 0;
		std::chrono::steady_clock::time_point mapping_start, mapping_end, loading_start, loading_end, preview_start, preview_end;)
	while (cv::waitKey(1) != 27)
	{
		TIMES(loading_start = std::chrono::steady_clock::now();)
		// get a new frame from camera
		(in.*fn)(frame);
		if (frame.empty())
			break;
		TIMES(loading_end = std::chrono::steady_clock::now();
			  loading_sum += std::chrono::duration_cast<std::chrono::duration<double>>(loading_end - loading_start).count();)

		TIMES(mapping_start = std::chrono::steady_clock::now();)
		// remap(frame, map, equiframe);
		remap(frame, frame(cv::Rect(map.rows, 0, map.rows, map.rows)), map, equiframe);
		TIMES(mapping_end = std::chrono::steady_clock::now();
			  mapping_sum += std::chrono::duration_cast<std::chrono::duration<double>>(mapping_end - mapping_start).count();)

		TIMES(preview_start = std::chrono::steady_clock::now();)
		// Auf den Schirm Spoki
		imshow(WINDOW_NAME, equiframe);
		// if (wrt.isOpened())
		// 	wrt << equiframe;
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
	}
}

template <typename T, typename ACC_FN>
void loop(T in_1, T in_2, ACC_FN fn, const cv::Mat &map, cv::Mat &equiframe)
{
	cv::Mat frame_1, frame_2;
	TIMES(
		unsigned int nrframes = 0; double mapping_sum = 0, loading_sum = 0, preview_sum = 0;
		std::chrono::steady_clock::time_point mapping_start, mapping_end, loading_start, loading_end, preview_start, preview_end;)
	while (cv::waitKey(1) != 27)
	{
		TIMES(loading_start = std::chrono::steady_clock::now();)
		// get a new frame from camera
		(in_1.*fn)(frame_1);
		(in_2.*fn)(frame_2);
		if (frame_1.empty() || frame_2.empty())
			break;
		TIMES(loading_end = std::chrono::steady_clock::now();
			  loading_sum += std::chrono::duration_cast<std::chrono::duration<double>>(loading_end - loading_start).count();)

		TIMES(mapping_start = std::chrono::steady_clock::now();)
		remap(frame_1, frame_2, map, equiframe);
		TIMES(mapping_end = std::chrono::steady_clock::now();
			  mapping_sum += std::chrono::duration_cast<std::chrono::duration<double>>(mapping_end - mapping_start).count();)

		TIMES(preview_start = std::chrono::steady_clock::now();)
		// Auf den Schirm Spoki
		imshow(WINDOW_NAME, equiframe);
		// if (wrt.isOpened())
		// 	wrt << equiframe;
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
	}
}

int main(int argc, char **argv)
{
	string map_file_path, vid_path_1, vid_path_2, out_path;
	cv::Mat mapping_table;

	parse_args(argc, argv, map_file_path, vid_path_1, vid_path_2, out_path);

	read_mapping_file(map_file_path, mapping_table);

	cv::VideoCapture cap_1, cap_2;
	cv::Mat frame_1, frame_2, equiframe;

	TIMES(
		unsigned int nrframes = 0; double mapping_sum = 0, loading_sum = 0, preview_sum = 0;
		std::chrono::steady_clock::time_point mapping_start, mapping_end, loading_start, loading_end, preview_start, preview_end;)

	// create windows
	cv::namedWindow(WINDOW_NAME, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
	cv::resizeWindow(WINDOW_NAME, 1440, 720);

	// init output mat
	equiframe.create(mapping_table.rows, mapping_table.cols, CV_8UC3);

	//TODO do better implementation
	cv::VideoWriter wrt;

	if (vid_path_2.empty()) // one input
	{
		// open video
		cap_1.open(vid_path_1);
		if (!cap_1.isOpened())
		{
			cout << "Could not open video file \"" << vid_path_1 << "\"." << endl;
			return EXIT_FAILURE;
		}

		// get first frame and check if file is empty
		cap_1 >> frame_1;
		if (frame_1.empty())
		{
			cout << "Video file \"" << vid_path_1 << "\" is empty." << endl;
			return EXIT_FAILURE;
		}

		// check resolution
		if (frame_1.cols != mapping_table.cols || frame_1.rows != mapping_table.rows)
		{
			cout << "Image(" << frame_1.cols << 'x' << frame_1.rows
				 << ") and map(" << mapping_table.cols << 'x' << mapping_table.rows << ") resolution don't match." << endl;
			return EXIT_FAILURE;
		}

		// process single image
		cap_1 >> frame_2;
		if (frame_2.empty())
		{
			remap(frame_1, mapping_table, equiframe);
			cv::imshow(WINDOW_NAME, equiframe);
			if (!out_path.empty())
				cv::imwrite(out_path, equiframe);
			cv::waitKey();
			return EXIT_SUCCESS;
		}

		// jump back to begining
		cap_1.set(cv::CAP_PROP_POS_FRAMES, 0);

		// TODO open video output
		if (!out_path.empty())
			wrt.open(out_path, cap_1.get(cv::CAP_PROP_FOURCC), cap_1.get(cv::CAP_PROP_FPS), cv::Size(mapping_table.cols, mapping_table.rows));

		// do mapping
		loop(cap_1, &cv::VideoCapture::read, mapping_table, equiframe);
		// while (cv::waitKey(1) != 27)
		// {
		// 	TIMES(mapping_start = std::chrono::steady_clock::now();)
		// 	remap(frame_1, mapping_table, equiframe);
		// 	TIMES(mapping_end = std::chrono::steady_clock::now();
		// 		  mapping_sum += std::chrono::duration_cast<std::chrono::duration<double>>(mapping_end - mapping_start).count();)

		// 	TIMES(preview_start = std::chrono::steady_clock::now();)
		// 	// Auf den Schirm Spoki
		// 	imshow(WINDOW_NAME, equiframe);
		// 	if (wrt.isOpened())
		// 		wrt << equiframe;
		// 	TIMES(preview_end = std::chrono::steady_clock::now();
		// 		  preview_sum += std::chrono::duration_cast<std::chrono::duration<double>>(preview_end - preview_start).count();)

		// 	TIMES(loading_start = std::chrono::steady_clock::now();)
		// 	// get a new frame from camera
		// 	cap_1 >> frame_1;
		// 	if (frame_1.empty())
		// 		break;
		// 	TIMES(loading_end = std::chrono::steady_clock::now();
		// 		  loading_sum += std::chrono::duration_cast<std::chrono::duration<double>>(loading_end - loading_start).count();)

		// 	// check time
		// 	TIMES(nrframes++;
		// 		  if ((nrframes % 300) == 0) {
		// 			  printf("frame nr: %u\n", nrframes);
		// 			  printf("%f avg mapping time\n", mapping_sum / nrframes);
		// 			  printf("%f avg loading time\n", loading_sum / nrframes);
		// 			  printf("%f avg preview time\n", preview_sum / nrframes);
		// 			  printf("%f fps mapping\n", nrframes / mapping_sum);
		// 			  printf("%f fps mapping + loading\n", nrframes / (mapping_sum + loading_sum));
		// 			  printf("%f fps mapping + loading + preview\n", nrframes / (mapping_sum + loading_sum + preview_sum));
		// 			  printf("\n");
		// 		  })
		// }
	}
	else // two inputs
	{
		// open video
		cap_1.open(vid_path_1);
		cap_2.open(vid_path_2);
		if (!cap_1.isOpened())
			cout << "Could not open video file \"" << vid_path_1 << "\"." << endl;
		if (!cap_2.isOpened())
			cout << "Could not open video file \"" << vid_path_1 << "\"." << endl;
		if (!cap_1.isOpened() || !cap_2.isOpened())
			return EXIT_FAILURE;

		// get first frame and check if file is empty
		cap_1 >> frame_1;
		cap_2 >> frame_2;
		if (frame_1.empty())
			cout << "Video file \"" << vid_path_1 << "\" is empty." << endl;
		if (frame_2.empty())
			cout << "Video file \"" << vid_path_2 << "\" is empty." << endl;
		if (frame_1.empty() || frame_2.empty())
			return EXIT_FAILURE;

		// check resolution
		if (frame_1.cols * 2 != mapping_table.cols || frame_1.rows != mapping_table.rows || frame_2.cols * 2 != mapping_table.cols || frame_2.rows != mapping_table.rows)
		{
			cout << "Image1(" << frame_1.cols << 'x' << frame_1.rows << "), Image2(" << frame_2.cols << 'x' << frame_2.rows
				 << ") and map(" << mapping_table.cols << 'x' << mapping_table.rows << ") resolution don't match." << endl;
			return EXIT_FAILURE;
		}

		// process single image
		if (!cap_1.grab())
		{
			remap(frame_1, frame_2, mapping_table, equiframe);
			cv::imshow(WINDOW_NAME, equiframe);
			if (!out_path.empty())
				cv::imwrite(out_path, equiframe);
			cv::waitKey();
			return EXIT_SUCCESS;
		}

		// jump back to begining
		cap_1.set(cv::CAP_PROP_POS_FRAMES, 0);
		cap_2.set(cv::CAP_PROP_POS_FRAMES, 0);

		if (!out_path.empty())
			wrt.open(out_path, cap_1.get(cv::CAP_PROP_FOURCC), cap_1.get(cv::CAP_PROP_FPS), cv::Size(mapping_table.cols, mapping_table.rows));

		// do mapping
		loop(cap_1, cap_2, &cv::VideoCapture::read, mapping_table, equiframe);
		// while (cv::waitKey(1) != 27)
		// {
		// 	TIMES(mapping_start = std::chrono::steady_clock::now();)
		// 	remap(frame_1, frame_2, mapping_table, equiframe);
		// 	TIMES(mapping_end = std::chrono::steady_clock::now();
		// 		  mapping_sum += std::chrono::duration_cast<std::chrono::duration<double>>(mapping_end - mapping_start).count();)

		// 	TIMES(preview_start = std::chrono::steady_clock::now();)
		// 	// Auf den Schirm Spoki
		// 	imshow(WINDOW_NAME, equiframe);
		// 	if (wrt.isOpened())
		// 		wrt << equiframe;
		// 	TIMES(preview_end = std::chrono::steady_clock::now();
		// 		  preview_sum += std::chrono::duration_cast<std::chrono::duration<double>>(preview_end - preview_start).count();)

		// 	TIMES(loading_start = std::chrono::steady_clock::now();)
		// 	// get a new frame from camera
		// 	cap_1 >> frame_1;
		// 	cap_2 >> frame_2;
		// 	if (frame_1.empty() || frame_2.empty())
		// 		break;
		// 	TIMES(loading_end = std::chrono::steady_clock::now();
		// 		  loading_sum += std::chrono::duration_cast<std::chrono::duration<double>>(loading_end - loading_start).count();)

		// 	// check time
		// 	TIMES(nrframes++;
		// 		  if ((nrframes % 300) == 0) {
		// 			  printf("frame nr: %u\n", nrframes);
		// 			  printf("%f avg mapping time\n", mapping_sum / nrframes);
		// 			  printf("%f avg loading time\n", loading_sum / nrframes);
		// 			  printf("%f avg preview time\n", preview_sum / nrframes);
		// 			  printf("%f fps mapping\n", nrframes / mapping_sum);
		// 			  printf("%f fps mapping + loading\n", nrframes / (mapping_sum + loading_sum));
		// 			  printf("%f fps mapping + loading + preview\n", nrframes / (mapping_sum + loading_sum + preview_sum));
		// 			  printf("\n");
		// 		  })
		// }
	}

	// When everything done, release the video capture object
	cap_1.release();
	cap_2.release();
	// Closes all the frames
	cv::destroyAllWindows();

	return 0;
}
