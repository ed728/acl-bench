#ifndef LAYER_H
#define LAYER_H

#include <arm_compute/runtime/CL/CLFunctions.h>
#include <arm_compute/runtime/NEON/NEFunctions.h>

#include "TestTensor.hpp"

#include "arm_compute/runtime/CL/CLScheduler.h"

#include <tuple>
#include <time.h>
#include <initializer_list>
#include <vector>

#define NEW_LAYER(L, v) Layer<CL ## L, NE ## L> v

/* Creates a layer with the specified type. 
 *
 * Abstracts running a CL and NEON layer using the same back-end data.
 */
template <class _L_CL, class _L_NE>
class Layer
{
public:
	/* Layer constructor. 
	 *
	 * Performs preliminary ACL layer set-up, by creating the Tensors.
	 *
	 * @param [in] i_[xyz] The dimensions of the image used.
	 * @param [in] w_[xyz] The dimensions of the kernel/weights.
	 *
	 * TODO: This REALLY needs a rework. Currently this will work only for Convolution,
	 * which is fine since that is what I am looking to benchmark,
	 * HOWEVER, there is no point to this class and constructor anymore and run
	 * should be made a static function. 
	 *
	 */
	Layer(size_t i_x, size_t i_y, size_t i_z, size_t w_x, size_t w_y, size_t w_z)
	:	m_i_shape(i_x, i_y, i_z)
	,	m_w_shape(w_x, w_y, w_z)
	{
		CLScheduler::get().default_init();
	}

	/* Runs the layers.
	 *
	 * Runs the available layers for the specified amount of iterations, and returns
	 * an array of time periods of how long each run took for each layer.
	 *
	 * The result array needs to be provided by the user and needs to accommodate
	 * a result for each layer per iteration.
	 *
	 * @param [in]  iters The amount of iterations to perform.
	 * @param [out] res   The resulting list of times.
	 * @param [in]  info  The layer-specific information needed to configure the layer. Must be set-up by the user.
	 */
	template<typename _Info, class ... _TensorList>
	void run(unsigned int iters, float *res, _Info&& info, _TensorList& ... ttl)
	{
		clock_t t;

		/* Allocate the buffers and configure the layer.
		 * We do this only once since in an actual use-case the size of the tensors
		 * will be constant. */
		configure<CLTensor, _L_CL>(info, ttl ...);
		configure<Tensor, _L_NE>(info, ttl ...);

		for (size_t i = 0; i < iters; ++i)
		{
			/* Randomize the data once for both layers so nothing can be cached in any way. */
			(ttl.randomize(), ...);
			
			/* Run NEON and write results. */
			t = clock();
			run_layer<Tensor, _L_NE>(fill<Tensor>(ttl) ...);
			t = clock() - t;
			res[i*2] = ((float)t)/CLOCKS_PER_SEC;

			/* Run CL and write results. */	
			t = clock();
			run_layer<CLTensor, _L_CL>(fill<CLTensor>(ttl) ...);
			t = clock() - t;
			res[i*2+1] = ((float)t)/CLOCKS_PER_SEC;
		}
	}

	/* Conversions. */
	operator _L_CL&() { return m_cl_layer; }
	operator _L_NE&() { return m_ne_layer; }

private:

	template<class _T, class _L, class ... _TensorList>
	void run_layer(_TensorList& ... ttl)
	{	
		/* Synchronize the data between CPU and GPU if we are using CL. */
		if (std::is_same<_L, _L_CL>::value)
		{
			CLScheduler::get().sync();
		}

		/* Run the thing. */
		((_L&)*this).run();
	}

	/* Due to issues with how the Tensor and CLTensor classes are implemented
	 * we need the following hacks to call the templated member function
	 * specializations.
	 */

	/* Copy over the tensor data.
	 * As input, output, and possibly weight, bias data can be changing at runtime,
	 * we do this each iteration for each layer. This will also be a fair comparison,
	 * as it will force the GPU mapping, which is a real bottleneck and the purpose
	 * of this benchmark. */
	template<class _T>
	TestTensor& fill(TestTensor& tt)
	{
		tt.fill<_T>();
		return tt;
	}

	/* We return the same thing just to put this into the configure call.
	 * Should be done more nicely later.
	 */
	template<class _T>
	TestTensor& ready(TestTensor& tt)
	{
		tt.set_ready<_T>();
		return tt;
	}

	/* Configures the layer with the given tensor list. */
	template<class _Tensor, class _Layer, typename _Info, typename ... _TensorList>
	void configure(_Info&& info, _TensorList& ... ttl)
	{
		static_cast<_Layer&>(*this).configure(&static_cast<_Tensor&>(ready<_Tensor>(ttl))..., info);

	}

	_L_CL m_cl_layer;
	_L_NE m_ne_layer;

	TensorShape m_i_shape, m_w_shape;
};
#endif /* LAYER_H */
