#pragma once
#include "easytf/base.h"
#include "easytf/operator.h"

namespace easytf
{
	class OP_Reverse : public Operator
	{
	public:
		//meta string
		const static std::string meta_unit_dim;
	public:
		OP_Reverse() = default;
		OP_Reverse(const uint32_t unit_dim);
		//init param
		virtual void init(const Param& param) override;
		//forward
		virtual void forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top) override;
	private:
		void naive_implement(const float32_t* src, float32_t* dst, const int32_t unit_dim,const int32_t count);
	private:
		uint32_t unit_dim = 1;
	};
}