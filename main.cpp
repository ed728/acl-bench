#include "Layer.hpp"
#include <stdio.h>

#define ITERATIONS 20
#define LAYERS 2 /* NEON & CL */

int main()
{
	/* Create the layer. */
	NEW_LAYER(ConvolutionLayer, conv)(5, 5, 1, 5, 5, 8);
	/* Buffer needs to accomodate 1 floats per layer per iteration. */
	float *buff = (float *)malloc(ITERATIONS * LAYERS * sizeof(float));
	/* We need to pad our data by 2 to accommodate our kernel size. */
	conv.run(ITERATIONS, buff, PadStrideInfo(1, 1, 2, 2));
	/* Print the results. */
	for (int i = 0; i < ITERATIONS; ++i)
	{
		printf("NEON: %f | CL: %f\n", buff[i*2], buff[i*2+1]);
	}
	return 0;
}
