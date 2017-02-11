#include "easytf/operators/op_fullconnect.h"
#include "easytf/easytf_assert.h"
#include "easytf/easytf_logger.h"

//meta string
const std::string easytf::OP_FullConnect::meta_input_dim = "fc_input_dim";
const std::string easytf::OP_FullConnect::meta_output_dim = "fc_output_dim";
const std::string easytf::OP_FullConnect::meta_w = "fc_w";
const std::string easytf::OP_FullConnect::meta_bias = "fc_bias";

//
easytf::OP_FullConnect::OP_FullConnect(const int32_t input_dim, const int32_t output_dim, const float32_t* w, const float* bias)
{
	easytf::Param param;
	param.put_item(easytf::OP_FullConnect::meta_input_dim, easytf::Any(input_dim));
	param.put_item(easytf::OP_FullConnect::meta_output_dim, easytf::Any(output_dim));
	param.put_item(easytf::OP_FullConnect::meta_w, easytf::Any(w, input_dim*output_dim));
	param.put_item(easytf::OP_FullConnect::meta_bias, easytf::Any(bias, output_dim));
	init(param);
}
//init
void easytf::OP_FullConnect::init(const Param& param)
{
	this->w = param.get_item(meta_w);
	this->bias = param.get_item(meta_bias);
}
//forward
void easytf::OP_FullConnect::forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top)
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
		naive_implement(bottom_size, bottom_data + i*bottom_size, top_size, top_data+i*top_size);
	}
}

void easytf::OP_FullConnect::naive_implement(const int32_t src_size, const float32_t* src, const int32_t dst_size, float32_t* dst)
{
	const float32_t* weight_data = this->w.as_float32_array();
	const float32_t* bias_data = this->bias.as_float32_array();
	for (int32_t i = 0; i < dst_size;i++)
	{
		float32_t sum = 0.0f;
		for (int32_t j = 0; j < src_size;j++)
		{
			sum += weight_data[j*dst_size + i] * src[j];
		}
		sum += bias_data[i];
		dst[i] = sum;
	}
}