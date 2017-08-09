#include "TestTensor.hpp"
#include <cstring>

TestTensor::TestTensor(size_t x, size_t y, size_t z)
:	m_shape(TensorShape(x, y, z))
		
{
	m_cl.allocator()->init(TensorInfo(m_shape, 1, TYPE_ACL));
	m_ne.allocator()->init(TensorInfo(m_shape, 1, TYPE_ACL));
}

TestTensor::TestTensor(TensorShape ts)
:	m_shape(ts)
{
	m_cl.allocator()->init(TensorInfo(m_shape, 1, TYPE_ACL));
	m_ne.allocator()->init(TensorInfo(m_shape, 1, TYPE_ACL));
}

TestTensor::~TestTensor()
{
	if (nullptr != m_random)
	{
		free(m_random);
	}
}

CLTensor& TestTensor::getCL()
{
	return m_cl;
}
		
Tensor& TestTensor::getNE()
{
	return m_ne;
}

template<>
void TestTensor::set_ready<CLTensor>()
{
	m_cl.allocator()->allocate();
	m_cl_ready = true;
}

template<>
void TestTensor::set_ready<Tensor>()
{
	m_ne.allocator()->allocate();
	m_ne_ready = true;
}

template<>
bool TestTensor::is_ready<CLTensor>()
{
	return m_cl_ready;
}

template<>
bool TestTensor::is_ready<Tensor>()
{
	return m_ne_ready;
}

template<>
void TestTensor::fill<CLTensor>(void *src)
{
	/* We need to map and unmap the OpenCL buffer. */
	m_cl.map(true);
	uint8_t *dst = m_cl.buffer();
	std::memcpy(dst, src, get_size());		
	m_cl.unmap();
}

template<>
void TestTensor::fill<Tensor>(void *src)
{
	uint8_t *dst = m_ne.buffer();
	std::memcpy(dst, src, get_size());		
}

void TestTensor::randomize()
{
	/* Copying float. Does not need to match the Tensor data type
	 * as we are just after random data. */
	if (nullptr != m_random)
	{
		free(m_random);
	}
	m_random = malloc(get_size());
	float r = 0.f;
	for (size_t i = 0; i < get_size(); i += sizeof(r))
	{
		r = std::rand();
		std::memcpy(&((char *)m_random)[i], &r, sizeof(r));
	}
}

size_t TestTensor::get_size()
{
	return m_shape.total_size() * data_size_from_type(TYPE_ACL);
}
