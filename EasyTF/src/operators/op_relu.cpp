#include "easytf/operators/op_relu.h"
#include "easytf/easytf_assert.h"
#include "easytf/easytf_logger.h"

//meta string
//None

//init
void easytf::OP_Relu::init(const Param& param)
{
}
//forward
void easytf::OP_Relu::forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top)
{
	easyAssert(bottom.size() == 1 && top.size() == 1, "size of bottom and top must be 1.");
	auto bottom_iter = bottom.begin();
	auto top_iter = top.begin();
	//check
	const int32_t bottom_size = bottom_iter->second->get_shape().get_full_size();
	const float32_t* bottom_data = bottom_iter->second->get_data().as_float32_array();
	const int32_t top_size = top_iter->second->get_shape().get_full_size();
	float32_t* top_data = top_iter->second->get_data().as_float32_array();
	easyAssert(bottom_size == top_size, "top_size must be equals with bottom_size.");
	easyAssert(bottom_data && top_data, "bottom_data and top_data can't be empty.");

	naive_implement(bottom_data, top_data, bottom_size);
}
void easytf::OP_Relu::naive_implement(const float32_t* src, float32_t* dst, const int32_t size)
{
	for (int32_t i = 0; i < size; i++)
	{
		dst[i] = std::max(0.0f,src[i]);
	}
}