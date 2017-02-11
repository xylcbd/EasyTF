#include "easytf/param.h"
#include "easytf/easytf_assert.h"

easytf::Param::Param()
{

}
easytf::Param::Param(const std::map<std::string, Any>& items)
{
	this->items = items;
}
void easytf::Param::put_item(const std::string& meta, const Any& data)
{
	this->items[meta] = data;
}
const easytf::Any& easytf::Param::get_item(const std::string& meta) const
{
	auto iter = items.find(meta);
	easyAssert(iter != items.cend(), "item is not found.");
	return iter->second;
}