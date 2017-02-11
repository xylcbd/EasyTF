#include "easytf/operators/op_reverse.h"
#include "easytf/easytf_assert.h"
#include "easytf/easytf_logger.h"
//meta string
const std::string easytf::OP_Reverse::meta_unit_dim = "reverse_unit_dim";

easytf::OP_Reverse::OP_Reverse(const uint32_t unit_dim)
{
	Param param;
	param.put_item(meta_unit_dim, Any(unit_dim));
	init(param);
}
//init
void easytf::OP_Reverse::init(const Param& param)
{
	this->unit_dim = param.get_item(meta_unit_dim).as_int32();
}
//forward
void easytf::OP_Reverse::forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top)
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

	naive_implement(bottom_data, top_data, unit_dim, bottom_size / unit_dim);
}
void easytf::OP_Reverse::naive_implement(const float32_t* src, float32_t* dst, const int32_t unit_dim, const int32_t count)
{
	//unit_0 unit_1 unit_2 ... unit_N
	//unit_N unit_N-1 unit_N-2 ... unit_0
	for (int32_t i = 0; i < count; i++)
	{
		const float32_t* src_unit = src + i*unit_dim;
		float32_t* dst_unit = dst + (count - 1 - i)*unit_dim;
		memcpy(dst_unit, src_unit, unit_dim*sizeof(float32_t));
	}
}