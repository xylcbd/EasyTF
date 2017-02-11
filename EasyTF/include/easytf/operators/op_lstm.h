#pragma once
#include "easytf/base.h"
#include "easytf/operators/op_recurrent.h"
#include "easytf/operators/op_elementwise.h"
#include "easytf/operators/op_sigmoid.h"
#include "easytf/operators/op_hardsigmoid.h"
#include "easytf/operators/op_tanh.h"

namespace easytf
{
	class OP_LSTM : public OP_Recurrent
	{
	public:
		OP_LSTM() = default;
		OP_LSTM(const int32_t input_dim, const int32_t output_dim, const bool output_sequence,
			const float32_t* wih, const float32_t* whh, const  float32_t* bias);
		//init param
		virtual void init(const Param& param) override;
		//forward
		virtual void forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top) override;		
	private:
		void naive_implement(const int32_t src_frames, const int32_t src_dim, const float32_t* src, const int32_t dst_frames, const int32_t dst_dim, float32_t* dst);
	private:
		OP_Mul op_mul;
		OP_Add op_add;
		OP_HardSigmoid op_sigmoid;
		OP_Tanh op_tanh;
	};
}
