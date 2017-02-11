#include "easytf/operators/OP_Recurrent.h"
#include "easytf/easytf_assert.h"
#include "easytf/easytf_logger.h"

//meta string
const std::string easytf::OP_Recurrent::meta_input_dim = "recurrent_input_dim";
const std::string easytf::OP_Recurrent::meta_output_dim = "recurrent_output_dim";
const std::string easytf::OP_Recurrent::meta_output_sequence = "recurrent_output_sequence";
const std::string easytf::OP_Recurrent::meta_wih = "recurrent_wih";
const std::string easytf::OP_Recurrent::meta_whh = "recurrent_whh";
const std::string easytf::OP_Recurrent::meta_bias = "recurrent_bias";

//init
void easytf::OP_Recurrent::init(const Param& param)
{
	this->input_dim = param.get_item(meta_input_dim).as_int32();
	this->output_dim = param.get_item(meta_output_dim).as_int32();
	this->output_sequence = param.get_item(meta_output_sequence).as_bool();
	this->w_ih = param.get_item(meta_wih);
	this->w_hh = param.get_item(meta_whh);
	this->bias = param.get_item(meta_bias);
	//cache : one frame
	this->cache = Any((float32_t*)nullptr, this->output_dim);
	this->cache.fill_zero();
}
//forward
void easytf::OP_Recurrent::forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top)
{
	easyAssert(bottom.size() == 1 && top.size() == 1, "size of bottom and top must be 1.");
	auto bottom_iter = bottom.begin();
	auto top_iter = top.begin();
	//check
	const Shape bottom_shape = bottom_iter->second->get_shape();
	const int32_t bottom_batch = bottom_shape.get_item(0);
	const int32_t bottom_frames = bottom_shape.get_item(1);
	const int32_t bottom_dim = bottom_shape.get_item(2);
	const int32_t bottom_size = bottom_frames*bottom_dim;
	const float32_t* bottom_data = bottom_iter->second->get_data().as_float32_array();

	const Shape top_shape = top_iter->second->get_shape();
	const int32_t top_batch = top_shape.get_item(0);
	const int32_t top_frames = top_shape.get_item(1);
	const int32_t top_dim = top_shape.get_item(2);
	const int32_t top_size = top_frames*top_dim;
	top_iter->second->get_data().fill_zero();
	float32_t* top_data = top_iter->second->get_data().as_float32_array();
	if (this->output_sequence)
	{
		easyAssert(bottom_frames == top_frames, "top_frames must be equals with bottom_frames , because of output_sequence is true."); 
	}
	else
	{
		easyAssert(top_frames == 1, "top_frames must be 1 , because of output_sequence is false.");
	}
	easyAssert(bottom_batch == top_batch, "bottom_batch must be equals with top_batch.");
	easyAssert(bottom_data && top_data, "bottom_data and top_data can't be empty.");

	for (int32_t i = 0; i < bottom_batch; i++)
	{
		const int32_t bottom_offset = i*bottom_size;
		const int32_t top_offset = i*top_size;
		naive_implement(bottom_frames, bottom_dim, bottom_data + bottom_offset, top_frames, top_dim, top_data + top_offset);
	}
}
void easytf::OP_Recurrent::naive_implement(const int32_t src_frames, const int32_t src_dim, const float32_t* src, const int32_t dst_frames, const int32_t dst_dim, float32_t* dst)
{
	const float32_t* wih_data = this->w_ih.as_float32_array();
	const float32_t* whh_data = this->w_hh.as_float32_array();
	const float32_t* bias_data = this->bias.as_float32_array();
	this->cache.fill_zero();
	float32_t* pre_time_data = this->cache.as_float32_array();

	for (int t = 0; t < src_frames;t++)
	{
		const float32_t* cur_input = src + t*src_dim;
		const float32_t* pre_output = nullptr;		
		float32_t* cur_output = nullptr;
		if (this->output_sequence)
		{
			pre_output = (t==0) ? pre_time_data : (dst + (t - 1)*dst_dim);
			cur_output = dst + t*dst_dim;
		}
		else
		{
			pre_output = pre_time_data;
			cur_output = dst;
		}
		//cur_output = wih * cur_input + whh*pre_output + bias			
		for (int i = 0; i < dst_dim;i++)
		{
			float32_t sum = 0.0f;
			// sum += wih * cur_input
			for (int j = 0; j < src_dim;j++)
			{
				sum += wih_data[j*dst_dim + i] * cur_input[j];
			}
			// sum += whh * pre_output
			for (int j = 0; j < dst_dim; j++)
			{
				sum += whh_data[i*dst_dim + j] * pre_output[j];
			}
			// sum += bias
			sum += bias_data[i];
		}
		if (!this->output_sequence)
		{
			memcpy(pre_time_data, cur_output, dst_dim*sizeof(float32_t));
		}
	}
}