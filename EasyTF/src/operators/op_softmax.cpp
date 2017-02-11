#include "easytf/operators/op_softmax.h"
#include "easytf/easytf_assert.h"
#include "easytf/easytf_logger.h"

//meta string
//None

//init
void easytf::OP_Softmax::init(const Param& param)
{
}
//forward
void easytf::OP_Softmax::forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top)
{
	easyAssert(bottom.size() == 1 && top.size() == 1, "size of bottom and top must be 1.");
	auto bottom_iter = bottom.begin();
	auto top_iter = top.begin();
	//check
	const int32_t bottom_batch = bottom_iter->second->get_shape().get_item(0);
	const int32_t bottom_size = bottom_iter->second->get_shape().get_full_size() / bottom_batch;
	const float32_t* bottom_data = bottom_iter->second->get_data().as_float32_array();
	const int32_t top_batch = bottom_iter->second->get_shape().get_item(0);
	const int32_t top_size = top_iter->second->get_shape().get_full_size() / top_batch;
	float32_t* top_data = top_iter->second->get_data().as_float32_array();
	easyAssert(bottom_batch == top_batch, "bottom_batch must be equals with top_batch.");
	easyAssert(bottom_size == top_size, "top_size must be equals with bottom_size.");
	easyAssert(bottom_data && top_data, "bottom_data and top_data can't be empty.");

	for (int32_t i = 0; i < bottom_batch; i++)
	{
		implement(bottom_data + i*bottom_size, top_data + i*bottom_size, bottom_size);
	}
}
//for other's usage
void easytf::OP_Softmax::implement(const float32_t* src, float32_t* dst, const int32_t size)
{
	naive_implement(src, dst, size);
}
void easytf::OP_Softmax::naive_implement(const float32_t* src, float32_t* dst, const int32_t size)
{
	//step1 : find max value
	float32_t maxVal = src[0];
	for (int32_t i = 0; i < size; i++)
	{
		maxVal = std::max(maxVal, src[i]);
	}
	//step2 : sum
	float32_t sum = 0;
	for (int32_t i = 0; i < size; i++)
	{
		dst[i] = std::exp(src[i] - maxVal);
		sum += dst[i];
	}
	//step3 : div
	for (int32_t i = 0; i < size; i++)
	{
		dst[i] = dst[i] / sum;
	}
}