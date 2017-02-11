#include "easytf/entity.h"

easytf::Entity::Entity(const Shape shape)
{
	this->shape = shape;
	this->data = Any((float32_t*)nullptr,shape.get_full_size());
}