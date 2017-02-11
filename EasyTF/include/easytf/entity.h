#pragma once
#include "easytf/base.h"
#include "easytf/shape.h"
#include "easytf/any.h"

namespace easytf
{
	class Entity
	{
	public:
		Entity(const Shape shape);
	public:
		inline const Shape get_shape() const { return shape; }
		inline const Any& get_data() const { return data; }
		inline Any& get_data() { return data; }
	private:
		Shape shape;
		Any data;
	};
}