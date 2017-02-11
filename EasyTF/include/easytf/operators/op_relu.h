#pragma once
#include "easytf/base.h"
#include "easytf/operator.h"

namespace easytf
{
	class OP_Relu : public Operator
	{
	public:
		//meta string
		//None
	public:
		//init param
		virtual void init(const Param& param) override;
		//forward
		virtual void forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top) override;
	private:
		//None
		static void naive_implement(const float32_t* src, float32_t* dst, const int32_t size);
	};
}