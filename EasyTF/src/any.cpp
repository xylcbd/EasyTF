#include "easytf/any.h"
#include "easytf/easytf_assert.h"

easytf::Any::Any()
{
	//no op
}
easytf::Any::Any(const bool data)
{
	holder.resize(sizeof(data));
	memcpy(&holder[0], &data, holder.size());
}
easytf::Any::Any(const int32_t data)
{
	holder.resize(sizeof(data));
	memcpy(&holder[0], &data, holder.size());
}
easytf::Any::Any(const uint32_t data)
{
	holder.resize(sizeof(data));
	memcpy(&holder[0], &data, holder.size());
}
easytf::Any::Any(const float32_t data)
{
	holder.resize(sizeof(data));
	memcpy(&holder[0], &data, holder.size());
}
easytf::Any::Any(const int32_t* data, const uint32_t size)
{
	easyAssert(size > 0, "parameter invalidate.");
	holder.resize(size*sizeof(data[0]));
	if (data)
	{
		memcpy(&holder[0], data, holder.size());
	}
}
easytf::Any::Any(const uint32_t* data, const uint32_t size)
{
	easyAssert(size > 0, "parameter invalidate.");
	holder.resize(size*sizeof(data[0]));
	if (data)
	{
		memcpy(&holder[0], data, holder.size());
	}
}
easytf::Any::Any(const float32_t* data, const uint32_t size)
{
	easyAssert(size > 0, "parameter invalidate.");
	holder.resize(size*sizeof(data[0]));
	if (data)
	{
		memcpy(&holder[0], data, holder.size());
	}
}
void easytf::Any::fill_zero()
{
	easyAssert(holder.size() > 0, "holder is invalidate.");
	memset(&holder[0], 0, holder.size());
}
bool easytf::Any::as_bool() const
{
	easyAssert(holder.size() > 0, "holder is invalidate.");
	return *(bool*)&holder[0];
}
int32_t easytf::Any::as_int32() const
{
	easyAssert(holder.size() >= sizeof(int32_t), "holder is invalidate.");
	return *(int32_t*)&holder[0];
}
uint32_t easytf::Any::as_uint32() const
{
	easyAssert(holder.size() >= sizeof(uint32_t), "holder is invalidate.");
	return *(int32_t*)&holder[0];
}
float32_t easytf::Any::as_float32() const
{
	easyAssert(holder.size() >= sizeof(float32_t), "holder is invalidate.");
	return *(float32_t*)&holder[0];
}
int32_t* easytf::Any::as_int32_array() const
{
	easyAssert(holder.size() >= sizeof(int32_t), "holder is invalidate.");
	return (int32_t*)&holder[0];
}
uint32_t* easytf::Any::as_uint32_array() const
{
	easyAssert(holder.size() >= sizeof(uint32_t), "holder is invalidate.");
	return (uint32_t*)&holder[0];
}
float32_t* easytf::Any::as_float32_array() const
{
	easyAssert(holder.size() >= sizeof(float32_t), "holder is invalidate.");
	return (float32_t*)&holder[0];
}
