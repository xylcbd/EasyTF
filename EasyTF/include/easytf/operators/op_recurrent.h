#pragma once
#include "easytf/base.h"
#include "easytf/operator.h"

namespace easytf
{
	//TODO : cover variable length in batch
	class OP_Recurrent : public Operator
	{
	public:
		//meta string
		const static std::string meta_input_dim;
		const static std::string meta_output_dim;
		const static std::string meta_output_sequence;
		const static std::string meta_wih;
		const static std::string meta_whh;
		const static std::string meta_bias;
	public:
		//init param
		virtual void init(const Param& param) override;
		//forward
		virtual void forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top) override;
	private:
		void naive_implement(const int32_t src_frames, const int32_t src_dim, const float32_t* src, const int32_t dst_frames, const int32_t dst_dim, float32_t* dst);
	protected:
		//input_dim
		int32_t input_dim = 0;
		//output_dim
		int32_t output_dim = 0;
		//if output sequence : for stack recurrent layer
		bool output_sequence = false;
		//weight : input -> hidden
		Any w_ih;
		//weight : hidden -> hidden
		Any w_hh;
		//bias
		Any bias;
		//cache
		Any cache;
	};
}
