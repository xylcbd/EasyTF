#include "easytf/operators/op_elementwise.h"
#include "easytf/easytf_assert.h"
#include "easytf/easytf_logger.h"

//meta string
//None

//init
void easytf::OP_Elementwise::init(const Param& param)
{
}
//forward
void easytf::OP_Elementwise::forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top)
{
	easyAssert(bottom.size() == 2 && top.size() == 1, "size of bottom must be 2, and size of top must be 1.");
	auto bottom_iter = bottom.begin();
	auto bottom_iter2 = ++bottom_iter; bottom_iter--;
	auto top_iter = top.begin();
	//check
	const int32_t bottom_size = bottom_iter->second->get_shape().get_full_size();
	const float32_t* bottom_data = bottom_iter->second->get_data().as_float32_array();
	const int32_t bottom_size2 = bottom_iter2->second->get_shape().get_full_size();
	const float32_t* bottom_data2 = bottom_iter2->second->get_data().as_float32_array();
	const int32_t top_size = top_iter->second->get_shape().get_full_size();
	float32_t* top_data = top_iter->second->get_data().as_float32_array();
	easyAssert(bottom_size == bottom_size2, "bottom_size2 must be equals with bottom_size.");
	easyAssert(bottom_size == top_size, "top_size must be equals with bottom_size.");
	easyAssert(bottom_data && bottom_data && top_data, "bottom_data and top_data can't be empty.");

	implement(bottom_data, bottom_data2,top_data, bottom_size);
}

//Mul
void easytf::OP_Mul::implement(const float32_t* src1, const float32_t* src2, float32_t* dst, const int32_t size)
{
	this->naive_implement(src1, src2, dst, size);
}
void easytf::OP_Mul::naive_implement(const float32_t* src1, const float32_t* src2, float32_t* dst, const int32_t size)
{
	for (int32_t i = 0; i < size; i++)
	{
		dst[i] = src1[i] * src2[i];
	}
}
void easytf::OP_Mul::implement(const float32_t* src1, const float32_t val, float32_t* dst, const int32_t size)
{
	this->naive_implement(src1, val, dst, size);
}
void easytf::OP_Mul::naive_implement(const float32_t* src1, const float32_t val, float32_t* dst, const int32_t size)
{
	for (int32_t i = 0; i < size;i++)
	{
		dst[i] = src1[i] * val;
	}
}

//Add
void easytf::OP_Add::implement(const float32_t* src1, const float32_t* src2, float32_t* dst, const int32_t size)
{
	this->naive_implement(src1, src2, dst, size);
}
void easytf::OP_Add::naive_implement(const float32_t* src1, const float32_t* src2, float32_t* dst, const int32_t size)
{
	for (int32_t i = 0; i < size; i++)
	{
		dst[i] = src1[i] + src2[i];
	}
}
void easytf::OP_Add::implement(const float32_t* src1, const float32_t val, float32_t* dst, const int32_t size)
{
	this->naive_implement(src1, val, dst, size);
}
void easytf::OP_Add::naive_implement(const float32_t* src1, const float32_t val, float32_t* dst, const int32_t size)
{
	for (int32_t i = 0; i < size; i++)
	{
		dst[i] = src1[i] + val;
	}
}