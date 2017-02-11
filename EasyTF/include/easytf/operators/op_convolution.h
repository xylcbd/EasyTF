#pragma once
#include "easytf/base.h"
#include "easytf/operator.h"

namespace easytf
{
	class OP_Convolution : public Operator
	{
	public:
		//meta string
		//shape
		const static std::string meta_kernel_num;
		const static std::string meta_kernel_channel;
		const static std::string meta_kernel_width;
		const static std::string meta_kernel_height;
		const static std::string meta_padding_width;
		const static std::string meta_padding_height;
		//data
		const static std::string meta_kernel;
		const static std::string meta_bias;
	public:
		OP_Convolution() = default;
		OP_Convolution(const int32_t kernel_num, const int32_t kernel_channel, 
			const int32_t kernel_width, const int32_t kernel_height,
			const int32_t padding_width, const int32_t padding_height,
			const float32_t* kernel, const float* bias);
		//init param
		virtual void init(const Param& param) override;
		//forward
		virtual void forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top) override;
	private:
		//None
		void naive_implement(const int32_t src_size, const float32_t* src, const int32_t dst_size, float32_t* dst);
	private:
		//weight
		Any kernel;
		//bias
		Any bias;
	};
}