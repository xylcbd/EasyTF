#include "easytf/shape.h"
#include "easytf/easytf_assert.h"

easytf::Shape::Shape()
{

}
easytf::Shape::Shape(const std::vector<int32_t> data)
{
	dataset = data;
}
int32_t easytf::Shape::get_item(const int32_t pos) const
{
	easyAssert(pos >= 0 && pos < (int32_t)(dataset.size()), "parameter is invalidate.");
	return dataset[pos];
}
uint32_t easytf::Shape::get_full_size() const
{
	easyAssert(dataset.size() > 0, "state is invalidate.");
	uint32_t size = 1;
	for (const auto& item : dataset)
	{
		size *= item;
	}
	return size;
}