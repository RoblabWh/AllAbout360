#include <iostream>
#include "mapping.h"

using namespace std;

int main(int argc, char **argv)
{
	if (argc < 3 || argc > 4)
	{
		cout << "Usage: " << argv[0] << " input1 [input2] map" << endl;
		return EXIT_FAILURE;
	}

	mapper *mapr;
	if (argc == 3)
		mapr = new mapper(argv[1], argv[2], 0, interpolation_type::BILINEAR);
	else
		mapr = new mapper(argv[1], argv[2], argv[3], 30, interpolation_type::NEAREST_NEIGHBOUR);

	cv::Mat img;
	TIMES( chrono::steady_clock::time_point lo_s, lo_e; double lo_sum; long fc = 0; )

	cv::namedWindow("Remapped", cv::WINDOW_KEEPRATIO);
	cv::resizeWindow("Remapped", 1440, 720);

	bool good = true;
	while (cv::waitKey(1) < 0 && good)
	{
		TIMES( lo_s = chrono::steady_clock::now(); )
		good = mapr->get_next_img(img);
		DBG( cout << "main good: " << boolalpha << good << '\n' << endl; )
		if (good)
		{
			cv::imshow("Remapped", img);
		}
		TIMES( lo_e = chrono::steady_clock::now(); lo_sum += std::chrono::duration_cast<std::chrono::duration<double>>(lo_e - lo_s).count(); )
		TIMES( fc++; if (fc % 150 == 0) {printf("\nmain frame: %ld\nloop=%fms, %ffps", fc, lo_sum / fc * 1000, fc / lo_sum);} )
	}
	
	delete mapr;
	DBG( cout << "mapper successfully deleted" << endl; )
	return EXIT_SUCCESS;
}
