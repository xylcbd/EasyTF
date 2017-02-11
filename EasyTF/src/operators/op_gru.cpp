#include "easytf/operators/op_gru.h"
#include "easytf/operators/op_sigmoid.h"
#include "easytf/operators/op_tanh.h"
#include "easytf/easytf_assert.h"
#include "easytf/easytf_logger.h"

easytf::OP_GRU::OP_GRU(const int32_t input_dim, const int32_t output_dim, const bool output_sequence,
	const float32_t* wih, const float32_t* whh, const float32_t* bias)
{
	easytf::Param param;
	param.put_item(easytf::OP_GRU::meta_input_dim, easytf::Any(input_dim));
	param.put_item(easytf::OP_GRU::meta_output_dim, easytf::Any(output_dim));
	param.put_item(easytf::OP_GRU::meta_output_sequence, easytf::Any(output_sequence));
	param.put_item(easytf::OP_GRU::meta_wih, easytf::Any(wih, 3 * input_dim*output_dim));
	param.put_item(easytf::OP_GRU::meta_whh, easytf::Any(whh, 3 * output_dim*output_dim));
	param.put_item(easytf::OP_GRU::meta_bias, easytf::Any(bias, 3 * output_dim));
	init(param);
}
//init
void easytf::OP_GRU::init(const Param& param)
{
	OP_Recurrent::init(param);
	//cache : two frame
	this->cache = Any((float32_t*)nullptr, 7 * this->output_dim);
	this->cache.fill_zero();
}
//forward
void easytf::OP_GRU::forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top)
{
	easyAssert(bottom.size() == 1 && top.size() == 1, "size of bottom and top must be 1.");
	auto bottom_iter = bottom.begin();
	auto top_iter = top.begin();
	//check
	const Shape bottom_shape = bottom_iter->second->get_shape();
	const int32_t bottom_batch = bottom_shape.get_item(0);
	const int32_t bottom_frames = bottom_shape.get_item(1);
	const int32_t bottom_dim = bottom_shape.get_item(2);
	const float32_t* bottom_data = bottom_iter->second->get_data().as_float32_array();

	const Shape top_shape = top_iter->second->get_shape();
	const int32_t top_batch = top_shape.get_item(0);
	const int32_t top_frames = top_shape.get_item(1);
	const int32_t top_dim = top_shape.get_item(2);
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
		naive_implement(bottom_frames, bottom_dim, bottom_data + i*bottom_frames*bottom_dim, top_frames, top_dim, top_data + i*top_frames*top_dim);
	}
}
void easytf::OP_GRU::naive_implement(const int32_t src_frames, const int32_t src_dim, const float32_t* src, const int32_t dst_frames, const int32_t dst_dim, float32_t* dst)
{
	//weight of input to hidden
	const float32_t* wih_data = this->w_ih.as_float32_array();
	const float32_t* wih_z_data = wih_data + 0 * src_dim * dst_dim;
	const float32_t* wih_r_data = wih_data + 1 * src_dim * dst_dim;
	const float32_t* wih_o_data = wih_data + 2 * src_dim * dst_dim;
	//weight of hidden to hidden
	const float32_t* whh_data = this->w_hh.as_float32_array();
	const float32_t* whh_z_data = whh_data + 0 * dst_dim * dst_dim;
	const float32_t* whh_r_data = whh_data + 1 * dst_dim * dst_dim;
	const float32_t* whh_o_data = whh_data + 2 * dst_dim * dst_dim;
	//weight of bias
	const float32_t* bias_data = this->bias.as_float32_array();
	const float32_t* bias_z_data = bias_data + 0 * dst_dim;
	const float32_t* bias_r_data = bias_data + 1 * dst_dim;
	const float32_t* bias_o_data = bias_data + 2 * dst_dim;

	this->cache.fill_zero();
	float32_t* cache_data = this->cache.as_float32_array();
	float32_t* pre_time_output_data = cache_data + 0 * dst_dim;
	float32_t* z_gate_data = cache_data + 1 * dst_dim;
	float32_t* r_gate_data = cache_data + 2 * dst_dim;
	float32_t* o_gate_data = cache_data + 3 * dst_dim;

	for (int32_t t = 0; t < src_frames; t++)
	{
		const float32_t* cur_input = src + t*src_dim;
		const float32_t* pre_output = nullptr;
		float32_t* cur_output = nullptr;
		if (this->output_sequence)
		{
			pre_output = (t == 0) ? pre_time_output_data : (dst + (t - 1)*dst_dim);
			cur_output = dst + t*dst_dim;
		}
		else
		{
			pre_output = pre_time_output_data;
			cur_output = dst;
		}

		//z_gate = sigmoid(wih_z * cur_input + whh_z * pre_output + bias_z)
		for (int32_t i = 0; i < dst_dim; i++)
		{
			float32_t sum = 0.0f;
			//sum += wih_i * cur_input
			for (int32_t j = 0; j < src_dim; j++)
			{
				sum += wih_z_data[j*dst_dim + i] * cur_input[j];
			}
			//sum += whh_i * pre_output
			for (int32_t j = 0; j < dst_dim; j++)
			{
				sum += whh_z_data[j*dst_dim + i] * pre_output[j];
			}
			//sum += bias_i
			sum += bias_z_data[i];
			z_gate_data[i] = sum;
		}
		op_sigmoid.implement(z_gate_data, z_gate_data, dst_dim);

		//r_gate = sigmoid(wih_r * cur_input + whh_r * pre_output + bias_r)
		for (int32_t i = 0; i < dst_dim; i++)
		{
			float32_t sum = 0.0f;
			//sum += wih_f * cur_input
			for (int32_t j = 0; j < src_dim; j++)
			{
				sum += wih_r_data[j*dst_dim + i] * cur_input[j];
			}
			//sum += whh_f * pre_output
			for (int32_t j = 0; j < dst_dim; j++)
			{
				sum += whh_r_data[j*dst_dim + i] * pre_output[j];
			}
			//sum += bias_f
			sum += bias_r_data[i];
			r_gate_data[i] = sum;
		}
		op_sigmoid.implement(r_gate_data, r_gate_data, dst_dim);

		//r_gate = r_gate * prev_output
		op_mul.implement(r_gate_data, pre_output, r_gate_data, dst_dim);
		//o_gate = tanh(wih_o * cur_input + whh_o * r_gate + bias_o)
		for (int32_t i = 0; i < dst_dim; i++)
		{
			float32_t sum = 0.0f;
			//sum += wih_o * cur_input
			for (int32_t j = 0; j < src_dim; j++)
			{
				sum += wih_o_data[j*dst_dim + i] * cur_input[j];
			}
			//sum += whh_o * pre_output
			for (int32_t j = 0; j < dst_dim; j++)
			{
				sum += whh_o_data[j*dst_dim + i] * r_gate_data[j];
			}
			//sum += bias_o
			sum += bias_o_data[i];
			o_gate_data[i] = sum;
		}
		op_tanh.implement(o_gate_data, o_gate_data, dst_dim);

		//z_gate = (1-z_gate)
		op_add.implement(z_gate_data, -1.0f, z_gate_data, dst_dim);
		//z_gate = z_gate*pre_output
		op_mul.implement(z_gate_data, pre_output, z_gate_data, dst_dim);

		//output = z_gate * o_gate
		op_mul.implement(z_gate_data, o_gate_data, cur_output, dst_dim);

		//output = output+z_gate
		op_add.implement(cur_output, z_gate_data, cur_output, dst_dim);

		if (!this->output_sequence)
		{
			memcpy(pre_time_output_data, cur_output, dst_dim*sizeof(float32_t));
		}
	}
}