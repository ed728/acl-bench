#ifndef TEST_TENSOR_HPP
#define TEST_TENSOR_HPP

#include "arm_compute/core/Types.h"
#include "utils/Utils.h"
#include "arm_compute/runtime/CL/CLTensor.h"

#include <stdlib.h>

using namespace arm_compute;
using namespace utils;

/* A class that creates a GPU and CPU Tensor with the same data.
 * 
 * This abstracts the functionality of having the same CL and NEON tensor
 * for benchmarking purposes.
 *
 * Provides the functionality for generating random weights and biases.
 */
class TestTensor
{
	public:
		/* Tensor constructor.
		 *
		 * Creates a Tensor with both GPU and CPU backed data.
		 *
		 * @param [in] w The width of the tensor.
		 * @param [in] h The height of the tensor.
		 * @param [in] z The depth of the tensor.
		 */
		TestTensor(size_t x, size_t y, size_t z);

		/* Tensor constructor.
		 *
		 * Creates a Tensor with both GPU and CPU backed data.
		 *
		 * @param [in] ts The shape of the Tensor.
		 */
		TestTensor(TensorShape ts);

		/* Destructor. */
		~TestTensor();

		/* Fills the tensor data.
		 *
		 * Fills the internal data structures with the provided data.
		 *
		 * The user must always allocate the buffer of the correct size, or suffer the consequences.
		 *
		 * Only works for CLTensor and Tensor classes.
		 *
		 * @param [in] data The data to fill the tensor with.
		 *
		 * @retval TRUE  The operation was successful.
		 * @retval FALSE Something failed. Quite likely the GPU mapping.
		 */
		template<class _T>
		void fill(void* data)
		{
		}

		/* Fills the tensor with random data.
		 *
		 * Fills the internal data structures with the randomized data. If randomize() has not been
		 * called, it will generate the random data.
		 *
		 * The user must always allocate the buffer of the correct size, or suffer the consequences.
		 *
		 * Only works for CLTensor and Tensor classes.
		 *
		 * @param [in] data The data to fill the tensor with.
		 *
		 * @retval TRUE  The operation was successful.
		 * @retval FALSE Something failed. Quite likely the GPU mapping.
		 */
		template<class _T>
		void fill()
		{
			if (nullptr == m_random)
			{
				randomize();
			}
			fill<_T>(m_random);
		}

		/* Creates an internal buffer with random data.
		 *
		 * This is used to later copy into each of the tensors and ensure they
		 * use the same data.
		 */
		void randomize();

		/* Sets the tensor to ready.
		 *
		 * This instructs the underlying allocator to actually allocate
		 * the data space, and needs to be called before any data can be written.
		 * This also records the state of the Tensor as ready.
		 *
		 * Currently work for CLTensor and Tensor classes, and does nothing
		 * for any other one.
		 */
		template<class _T>
		void set_ready()
		{
		}

		/* Whether the Tensor is ready.
		 *
		 * Returns whether the Tensor has been allocated and operations can already
		 * be performed on it.
		 *
		 * @retval TRUE  If Tensor is ready.
		 * @retval FALSE If Tensor is not ready or if it is not one of CLTensor and Tensor.
		 */
		template<class _T>
		bool is_ready()
		{
			return false;
		}

		/* Returns the underlying CL Tensor.
		 *
		 * @retval CL Tensor.
		 */
		CLTensor& getCL();
		
		/* Returns the underlying NEON Tensor.
		 *
		 * @retval NEON Tensor.
		 */
		Tensor& getNE();

		/* Returns the total size in bytes of the tensor.
		 *
		 * @retval Size of the tensor in bytes.
		 */
		size_t get_size();

		/* Conversions. */
		operator CLTensor&() { return m_cl; }
		operator Tensor&() { return m_ne; }

private:
		CLTensor m_cl;
		Tensor m_ne;

		bool m_cl_ready, m_ne_ready;
		TensorShape m_shape;
		void *m_random = nullptr;

		/* These need to be in sync. */
		const DataType TYPE_ACL = DataType::F32;
		typedef float TYPE_CPP ;
};
#endif /* TEST_TENSOR_HPP */
