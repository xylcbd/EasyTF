#pragma once
#include "easytf/base.h"
#include "easytf/operator.h"

namespace easytf
{
	class OP_FullConnect : public Operator
	{
	public:
		//meta string
		const static std::string meta_input_dim;
		const static std::string meta_output_dim;
		const static std::string meta_w;
		const static std::string meta_bias;
	public:
		OP_FullConnect() = default;
		OP_FullConnect(const int32_t input_dim,const int32_t output_dim,const float32_t* w, const float* bias);
		//init param
		virtual void init(const Param& param) override;
		//forward
		virtual void forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top) override;
	private:
		//None
		void naive_implement(const int32_t src_size, const float32_t* src, const int32_t dst_size,float32_t* dst);
	private:
		//weight
		Any w;
		//bias
		Any bias;
	};
}