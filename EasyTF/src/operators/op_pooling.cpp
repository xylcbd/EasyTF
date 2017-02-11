#include "easytf/operators/op_pooling.h"
#include "easytf/easytf_assert.h"
#include "easytf/easytf_logger.h"

//meta string
const std::string easytf::OP_Pooling::meta_type = "pooling_type";
const std::string easytf::OP_Pooling::meta_width = "pooling_width";
const std::string easytf::OP_Pooling::meta_height = "pooling_height";

//
easytf::OP_Pooling::OP_Pooling(const PoolingType type, const int32_t width, const int32_t height)
{
	easytf::Param param;
	param.put_item(easytf::OP_Pooling::meta_type, easytf::Any((int)type));
	param.put_item(easytf::OP_Pooling::meta_width, easytf::Any(width));
	param.put_item(easytf::OP_Pooling::meta_height, easytf::Any(height));
	init(param);
}
//init
void easytf::OP_Pooling::init(const Param& param)
{
	this->type = (PoolingType)(param.get_item(meta_type).as_int32());
	this->width = param.get_item(meta_width).as_int32();
	this->height = param.get_item(meta_height).as_int32();
}
//forward
void easytf::OP_Pooling::forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top)
{
	easyAssert(bottom.size() == 1 && top.size() == 1, "size of bottom and top must be 1.");
	auto bottom_iter = bottom.begin();
	auto top_iter = top.begin();
	//check
	const int32_t bottom_batch = bottom_iter->second->get_shape().get_item(0);
	const int32_t bottom_channel = bottom_iter->second->get_shape().get_item(1);
	const int32_t bottom_width = bottom_iter->second->get_shape().get_item(2);
	const int32_t bottom_height = bottom_iter->second->get_shape().get_item(3);
	const int32_t bottom_size = bottom_channel*bottom_width*bottom_height;
	const float32_t* bottom_data = bottom_iter->second->get_data().as_float32_array();
	const int32_t top_batch = bottom_iter->second->get_shape().get_item(0);
	const int32_t top_channel = bottom_iter->second->get_shape().get_item(1);
	const int32_t top_width = bottom_iter->second->get_shape().get_item(2);
	const int32_t top_height = bottom_iter->second->get_shape().get_item(3);
	const int32_t top_size = top_channel*top_width*top_height;
	float32_t* top_data = top_iter->second->get_data().as_float32_array();
	easyAssert(bottom_batch == top_batch, "bottom_batch must be equals with top_batch.");
	easyAssert(bottom_channel == top_channel, "bottom_channel must be equals with top_channel.");
	easyAssert(bottom_data && top_data, "bottom_data and top_data can't be empty.");
	for (int32_t i = 0; i < bottom_batch; i++)
	{
		naive_implement(bottom_channel, bottom_width, bottom_height,this->type,this->width,this->height,bottom_data + i*bottom_size, top_data + i*top_size);
	}
}

void easytf::OP_Pooling::naive_implement(const int32_t channel, const int32_t src_width, const int32_t src_height,
	const PoolingType type, const int32_t pooling_width, const int32_t pooling_height,
	const float32_t* src, float32_t* dst)
{
	//TODO
}