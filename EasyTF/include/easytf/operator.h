#pragma once
#include "easytf/base.h"
#include "easytf/entity.h"
#include "easytf/param.h"

namespace easytf
{
	class Operator
	{
	public:
		//init param
		virtual void init(const Param& param) = 0;
		//forward
		virtual void forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top) = 0;
	};
}