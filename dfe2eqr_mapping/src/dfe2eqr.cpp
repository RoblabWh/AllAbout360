#include <stdio.h>
#include <opencv2/opencv.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

// g++ -O2 -fopenmp -o dfe2eqr dfe2eqr.cpp  `pkg-config --cflags opencv` `pkg-config --libs opencv`

#ifdef DEBUG
#include <chrono>
#define DBG(E...) E
#else
#define DBG(E...)
#endif

int main(int argc, char **argv)
{
	FILE *fptr;
	int **mapx = NULL, **mapy = NULL;
	int height, width;
	int i, j;
	char *map_filename;
	bool is_single_input;

	// check for arguments
	if (!(argc == 3 || argc == 4))
	{
		printf("usage:\n");
		printf("\t%s <video> <mapping table>\n", argv[0]);
		printf("\t%s <video_front> <video_rear> <mapping table>\n", argv[0]);
		printf("example:\n");
		printf("\t%s example.mp4 mapping-table.txt\n", argv[0]);
		printf("\t%s example_front.mp4 example_rear.mp4 mapping-table.txt\n", argv[0]);
		return -1;
	}

	// read arguments
	if (argc == 3)
	{
		is_single_input = true;
		map_filename = argv[2];
		printf("Processing video file %s with mapping table %s \n", argv[1], map_filename);
	}
	else
	{
		is_single_input = false;
		map_filename = argv[3];
		printf("Processing video files %s and %s with mapping table %s \n", argv[1], argv[2], map_filename);
	}

	// read mapping table
	fptr = fopen(map_filename, "r");
	if (fptr != NULL)
	{
		if (fscanf(fptr, " %d %d \n", &width, &height) < 1)
			printf("Mapping file read error\n");
		printf("Open mapping file: h: %d w: %d\n", height, width);
		mapx = (int **)malloc(height * sizeof(int *));
		mapy = (int **)malloc(height * sizeof(int *));
		for (i = 0; i < height; i++)
		{
			mapx[i] = (int *)malloc(width * sizeof(int));
			mapy[i] = (int *)malloc(width * sizeof(int));
		}
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				if (fscanf(fptr, "%d %d\n", &(mapx[i][j]), &(mapy[i][j])) < 1)
				{
					printf("Mapping file read error at line %u\n", i * j);
					return -1;
				}
			}
		}
		fclose(fptr);
		printf("Mapping file %s read.\n", argv[2]);
	}
	else
		printf("Error opening mapping file %s\n", map_filename);

	// now read video
	cv::VideoCapture cap, cap_front, cap_rear;
	if (is_single_input)
	{
		cap.open(argv[1]);
		if (!cap.isOpened())
		{ // check if we succeeded
			printf("Could not open video file %s\n", argv[1]);
			return -1;
		}
	}
	else
	{
		cap_front.open(argv[1]);
		cap_rear.open(argv[2]);
		if (!cap_front.isOpened())
		{
			printf("Could not open video file %s\n", argv[1]);
			return -1;
		}
		if (!cap_rear.isOpened())
		{
			printf("Could not open video file %s\n", argv[2]);
			return -1;
		}
	}

	// create windows
	cv::namedWindow("frame", CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
	cv::resizeWindow("frame", 1440, 720);

	// create mats
	cv::Mat frame, frame_front, frame_rear;
	cv::Mat equiframe;

	DBG(unsigned int nrframes = 0; double mapping_sum = 0, loading_sum = 0, preview_sum = 0;
		std::chrono::steady_clock::time_point mapping_start, mapping_end, loading_start, loading_end, preview_start, preview_end;)

	// Get first frame and check if file is empty
	if (is_single_input)
	{
		cap >> frame;
		if (frame.empty())
		{
			printf("Video file is empty.\n");
			return -1;
		}
		frame_front = frame(cv::Rect(0, 0, height, height));
		frame_rear = frame(cv::Rect(height, 0, height, height));
	}
	else
	{
		cap_front >> frame_front;
		cap_rear >> frame_rear;
		if (frame_front.empty())
		{
			printf("Video file %s is empty.", argv[1]);
			return -1;
		}
		if (frame_rear.empty())
		{
			printf("Video file %s is empty.", argv[2]);
			return -1;
		}
	}
	// init output mat
	equiframe.create(height, width, frame_front.type());

	while (true)
	{
		DBG(mapping_start = std::chrono::steady_clock::now();)

//map to equirectangular
#pragma omp parallel for collapse(2) schedule(dynamic, 2048)
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				if (mapx[i][j] < height)
					equiframe.at<cv::Vec3b>(i, j) = frame_front.at<cv::Vec3b>(mapy[i][j], mapx[i][j]);
				else
					equiframe.at<cv::Vec3b>(i, j) = frame_rear.at<cv::Vec3b>(mapy[i][j], mapx[i][j] - height);
			}
		}

		DBG(mapping_end = std::chrono::steady_clock::now();
			mapping_sum += std::chrono::duration_cast<std::chrono::duration<double>>(mapping_end - mapping_start).count();)

		DBG(preview_start = std::chrono::steady_clock::now();)
		// Auf den Schirm Spoki
		imshow("frame", equiframe);
		DBG(preview_end = std::chrono::steady_clock::now();
			preview_sum += std::chrono::duration_cast<std::chrono::duration<double>>(preview_end - preview_start).count();)

		// check time
		DBG(nrframes++;
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

		// wait for esc to exit
		if (cv::waitKey(1) == 27)
			break;
		
		DBG(loading_start = std::chrono::steady_clock::now();)

		// get a new frame from camera
		if (is_single_input)
		{
			cap >> frame;
			if (frame.empty())
				break;
			frame_front = frame(cv::Rect(0, 0, height, height));
			frame_rear = frame(cv::Rect(height, 0, height, height));
		}
		else
		{
			cap_front >> frame_front;
			cap_rear >> frame_rear;
			if (frame_front.empty() || frame_rear.empty())
				break;
		}
		
		DBG(loading_end = std::chrono::steady_clock::now();
			loading_sum += std::chrono::duration_cast<std::chrono::duration<double>>(loading_end - loading_start).count();)
	}

	// When everything done, release the video capture object
	cap.release();
	// Closes all the frames
	cv::destroyAllWindows();

	return 0;
}
