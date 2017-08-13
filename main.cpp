#include "Layer.hpp"
#include <stdio.h>

#define ITERATIONS 10000
#define LAYERS 2 /* NEON & CL */

using namespace ACLBench;

/* Prints averages for each backend.
 * If the last parameter is set to true, also prints data for each run, which
 * can be used to write to file and then process the data later.
 */
void print_res(size_t sz, float *buff, bool raw = false)
{
	float count = sz/2.f;
	float tot_cl = 0.f;
	float tot_ne = 0.f;
	float avg_cl = 0.f;
	float avg_ne = 0.f;
	for (size_t i = 0; i < sz; ++i)
	{
		if (raw)
			printf("NEON: %f | CL: %f\n", buff[i*2], buff[i*2+1]);

		tot_cl += buff[i*2+1];
		tot_ne += buff[i*2];
	}
	avg_cl += tot_cl/count;
	avg_ne += tot_ne/count;
	if (raw)
		printf("---");
	printf("NEON: %f | CL: %f\n", avg_ne, avg_cl);

}

int main()
{
	/* Buffer needs to accomodate 1 floats per layer per iteration. */
	float *buff = (float *)malloc(ITERATIONS * LAYERS * sizeof(float));

	/* Procedure is:
	 * -create tensors;
	 * -RUN_LAYER;
	 * -print.
	 */

	/*** CONVOLUTION ***/
	TestTensor in(TensorShape(5, 5, 1));
	TestTensor weights(TensorShape(5, 5, 1, 8));
	TestTensor bias(TensorShape(8));
	TestTensor out(TensorShape(5, 5, 8));
	/* We need to pad our data by 2 to accommodate our kernel size. */
	RUN_LAYER(Convolution)(ITERATIONS, buff, PadStrideInfo(1, 1, 2, 2), in, weights, bias, out);
	printf("===CONVOLUTION===\n");
	print_res(ITERATIONS, buff);

	/*** RELU ***/
	TestTensor ac_in(TensorShape(20, 20, 8));
	TestTensor ac_out(TensorShape(20, 20, 8));
	RUN_LAYER(Activation)(ITERATIONS, buff,
	                      ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
	                      ac_in, ac_out);
	printf("===RELU===\n");
	print_res(ITERATIONS, buff);

	/* Clean-up. */
	free(buff);

	return 0;
}
