#pragma once
#include <map>
#include <string>
#include "easytf/base.h"
#include "easytf/any.h"

namespace easytf
{
	/*
	key-value data container
	*/
	class Param
	{
	public:
		Param();
		explicit Param(const std::map<std::string, Any>& items);
	public:
		void put_item(const std::string& meta, const Any& data);
		const Any& get_item(const std::string& meta) const;
	private:
		std::map<std::string, Any> items;
	};
}