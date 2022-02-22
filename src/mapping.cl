
__kernel void remap_nearest(__global unsigned char *g_idata_1, __global unsigned char *g_idata_2, __global unsigned char *g_odata, __global float *g_map, unsigned int len)
{
	unsigned int i = get_global_id(0);

	if (i < len)
	{
		unsigned int i3 = i * 3;
		unsigned int i4 = i * 4;
		int idx1 = *((int*)&(g_map[i4]));
		int idx2 = *((int*)&(g_map[i4 + 1]));
		float bf1 = g_map[i4 + 2];
		float bf2 = g_map[i4 + 3];

		g_odata[i3] = g_idata_1[idx1] * bf1 + g_idata_2[idx2] * bf2;
		g_odata[i3 + 1] = g_idata_1[idx1 + 1] * bf1 + g_idata_2[idx2 + 1] * bf2;
		g_odata[i3 + 2] = g_idata_1[idx1 + 2] * bf1 + g_idata_2[idx2 + 2] * bf2;
	}
}

__kernel void remap_linear(__global unsigned char *g_idata_1, __global unsigned char *g_idata_2, __global unsigned char *g_odata, __global float *g_map, unsigned int len, unsigned int step)
{
	unsigned int i = get_global_id(0);

	if (i < len)
	{
		unsigned int i3 = i * 3;
		unsigned int i10 = i * 10;
		int idx1 = *((int*)&(g_map[i10]));
		int idx2 = *((int*)&(g_map[i10 + 1]));

		unsigned int idx1_y1 = idx1 + step;
		unsigned int idx2_y1 = idx2 + step;
		unsigned int idx1_x1 = idx1 + 3;
		unsigned int idx2_x1 = idx2 + 3;
		unsigned int idx1_x1y1 = idx1_y1 + 3;
		unsigned int idx2_x1y1 = idx2_y1 + 3;

		float f11 = g_map[i10 + 2], f12 = g_map[i10 + 3],f13 = g_map[i10 + 4],f14 = g_map[i10 + 5];
		float f21 = g_map[i10 + 6], f22 = g_map[i10 + 7],f23 = g_map[i10 + 8],f24 = g_map[i10 + 9];

		g_odata[i3] = g_idata_1[idx1] * f11 + g_idata_1[idx1_x1] * f12 + g_idata_1[idx1_y1] * f13 + g_idata_1[idx1_x1y1] * f14
					+ g_idata_2[idx2] * f21 + g_idata_2[idx2_x1] * f22 + g_idata_2[idx2_y1] * f23 + g_idata_2[idx2_x1y1] * f24;
		g_odata[i3 + 1] = g_idata_1[idx1 + 1] * f11 + g_idata_1[idx1_x1 + 1] * f12 + g_idata_1[idx1_y1 + 1] * f13 + g_idata_1[idx1_x1y1 + 1] * f14
						+ g_idata_2[idx2 + 1] * f21 + g_idata_2[idx2_x1 + 1] * f22 + g_idata_2[idx2_y1 + 1] * f23 + g_idata_2[idx2_x1y1 + 1] * f24;
		g_odata[i3 + 2] = g_idata_1[idx1 + 2] * f11 + g_idata_1[idx1_x1 + 2] * f12 + g_idata_1[idx1_y1 + 2] * f13 + g_idata_1[idx1_x1y1 + 2] * f14
						+ g_idata_2[idx2 + 2] * f21 + g_idata_2[idx2_x1 + 2] * f22 + g_idata_2[idx2_y1 + 2] * f23 + g_idata_2[idx2_x1y1 + 2] * f24;
	}
}
