#pragma once
#include <vector>
#include "easytf/base.h"

namespace easytf
{
	class Shape
	{
	public:
		Shape();
		Shape(const std::vector<int32_t> data);
	public:
		int32_t get_item(const int32_t pos) const;
		inline int32_t get_size() const { return dataset.size(); }
		uint32_t get_full_size() const;
	private:
		std::vector<int32_t> dataset;
	};
}
