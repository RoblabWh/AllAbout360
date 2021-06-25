
__kernel void remap_nn(__global unsigned char *g_idata_1, __global unsigned char *g_idata_2, __global unsigned char *g_odata, __global double *g_map, unsigned int len)
{
	unsigned int i = get_global_id(0);

	if (i < len)
	{
		unsigned int i3 = i * 3;
		unsigned int i14 = i * 14;
		unsigned int idx1 = g_map[i14];
		unsigned int idx2 = g_map[i14 + 2];
		double bf1 = g_map[i14 + 4];
		double bf2 = g_map[i14 + 5];
		g_odata[i3] = g_idata_1[idx1] * bf1 + g_idata_2[idx2] * bf2;
		g_odata[i3 + 1] = g_idata_1[idx1 + 1] * bf1 + g_idata_2[idx2 + 1] * bf2;
		g_odata[i3 + 2] = g_idata_1[idx1 + 2] * bf1 + g_idata_2[idx2 + 2] * bf2;
	}
}

// __kernel void remap_nn(__global unsigned char3 *g_idata_1, __global unsigned char3 *g_idata_2, __global unsigned char3 *g_odata, __global double4 *g_map, unsigned int len)
// {
// 	unsigned int i = get_global_id(0);

// 	if (i < len)
// 	{
// 		g_odata[i] = g_idata_1[g_map[i][0]] * g_map[i][2] + g_idata_2[g_map[i][1]] * g_map[i][3];
// 	}
// }

__kernel void remap_li(__global unsigned char *g_idata_1, __global unsigned char *g_idata_2, __global unsigned char *g_odata, __global double *g_map, unsigned int len)
{
	unsigned int i = get_global_id(0);

	if (i < len)
	{
		unsigned int i3 = i * 3;
		unsigned int i14 = i * 14;
		unsigned int idx1 = g_map[i14];
		unsigned int idx2 = g_map[i14 + 2];
		double bf1 = g_map[i14 + 4];
		double bf2 = g_map[i14 + 5];
		unsigned int idx1_y1 = g_map[i14 + 1];
		unsigned int idx2_y1 = g_map[i14 + 3];
		unsigned int idx1_x1 = idx1 + 3;
		unsigned int idx2_x1 = idx2 + 3;
		unsigned int idx1_x1y1 = idx1_y1 + 3;
		unsigned int idx2_x1y1 = idx2_y1 + 3;
		double f11 = g_map[i14 + 6], f12 = g_map[i14 + 7],f13 = g_map[i14 + 8],f14 = g_map[i14 + 9];
		double f21 = g_map[i14 + 10], f22 = g_map[i14 + 11],f23 = g_map[i14 + 12],f24 = g_map[i14 + 13];

		g_odata[i3] = (g_idata_1[idx1] * f11 + g_idata_1[idx1_x1] * f12 + g_idata_1[idx1_y1] * f13 + g_idata_1[idx1_x1y1] * f14) * bf1
					+ (g_idata_2[idx2] * f21 + g_idata_2[idx2_x1] * f22 + g_idata_2[idx2_y1] * f23 + g_idata_2[idx2_x1y1] * f24) * bf2;
		g_odata[i3 + 1] = (g_idata_1[idx1 + 1] * f11 + g_idata_1[idx1_x1 + 1] * f12 + g_idata_1[idx1_y1 + 1] * f13 + g_idata_1[idx1_x1y1 + 1] * f14) * bf1
						+ (g_idata_2[idx2 + 1] * f21 + g_idata_2[idx2_x1 + 1] * f22 + g_idata_2[idx2_y1 + 1] * f23 + g_idata_2[idx2_x1y1 + 1] * f24) * bf2;
		g_odata[i3 + 2] = (g_idata_1[idx1 + 2] * f11 + g_idata_1[idx1_x1 + 2] * f12 + g_idata_1[idx1_y1 + 2] * f13 + g_idata_1[idx1_x1y1 + 2] * f14) * bf1
						+ (g_idata_2[idx2 + 2] * f21 + g_idata_2[idx2_x1 + 2] * f22 + g_idata_2[idx2_y1 + 2] * f23 + g_idata_2[idx2_x1y1 + 2] * f24) * bf2;
	}
}
