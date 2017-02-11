#include "easytf/operators/op_convolution.h"
#include "easytf/easytf_assert.h"
#include "easytf/easytf_logger.h"

//meta string
const std::string easytf::OP_Convolution::meta_kernel_num = "conv_kernel_num";
const std::string easytf::OP_Convolution::meta_kernel_channel = "conv_kernel_channel";
const std::string easytf::OP_Convolution::meta_kernel_width = "conv_kernel_width";
const std::string easytf::OP_Convolution::meta_kernel_height = "conv_kernel_height";
const std::string easytf::OP_Convolution::meta_padding_width = "conv_padding_width";
const std::string easytf::OP_Convolution::meta_padding_height = "conv_padding_height";

const std::string easytf::OP_Convolution::meta_kernel = "conv_kernel";
const std::string easytf::OP_Convolution::meta_bias = "conv_bias";

//
easytf::OP_Convolution::OP_Convolution(const int32_t kernel_num, const int32_t kernel_channel,
	const int32_t kernel_width, const int32_t kernel_height,
	const int32_t padding_width, const int32_t padding_height,
	const float32_t* kernel, const float* bias)
{
	easytf::Param param;
	param.put_item(easytf::OP_Convolution::meta_kernel_num, easytf::Any(kernel_num));
	param.put_item(easytf::OP_Convolution::meta_kernel_channel, easytf::Any(kernel_channel));
	param.put_item(easytf::OP_Convolution::meta_kernel_width, easytf::Any(kernel_width));
	param.put_item(easytf::OP_Convolution::meta_kernel_height, easytf::Any(kernel_height));
	param.put_item(easytf::OP_Convolution::meta_padding_width, easytf::Any(padding_width));
	param.put_item(easytf::OP_Convolution::meta_padding_height, easytf::Any(padding_height));

	param.put_item(easytf::OP_Convolution::meta_kernel, easytf::Any(kernel, kernel_num*kernel_channel*kernel_width*kernel_height));
	param.put_item(easytf::OP_Convolution::meta_bias, easytf::Any(bias, kernel_num));
	init(param);
}
//init
void easytf::OP_Convolution::init(const Param& param)
{
	this->kernel = param.get_item(meta_kernel);
	this->bias = param.get_item(meta_bias);
}
//forward
void easytf::OP_Convolution::forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top)
{
	easyAssert(bottom.size() == 1 && top.size() == 1, "size of bottom and top must be 1.");
	auto bottom_iter = bottom.begin();
	auto top_iter = top.begin();
	//check
	const int32_t bottom_batch = bottom_iter->second->get_shape().get_item(0);
	const int32_t bottom_size = bottom_iter->second->get_shape().get_full_size() / bottom_batch;
	const float32_t* bottom_data = bottom_iter->second->get_data().as_float32_array();
	const int32_t top_batch = bottom_iter->second->get_shape().get_item(0);
	const int32_t top_size = top_iter->second->get_shape().get_full_size() / bottom_batch;
	float32_t* top_data = top_iter->second->get_data().as_float32_array();
	easyAssert(bottom_batch == top_batch, "bottom_batch must be equals with top_batch.");
	easyAssert(bottom_data && top_data, "bottom_data and top_data can't be empty.");

	for (int32_t i = 0; i < bottom_batch; i++)
	{
		naive_implement(bottom_size, bottom_data + i*bottom_size, top_size, top_data + i*top_size);
	}
}

void easytf::OP_Convolution::naive_implement(const int32_t src_size, const float32_t* src, const int32_t dst_size, float32_t* dst)
{
	//TODO
}