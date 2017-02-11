#pragma once
#include "easytf/base.h"

namespace easytf
{
	/*
	like boost:any
	*/
	class Any
	{
	public:
		Any();
		explicit Any(const bool data);
		explicit Any(const int32_t data);
		explicit Any(const uint32_t data);
		explicit Any(const float32_t data);		
		//nullptr to allocate in Any
		explicit Any(const int32_t* data, const uint32_t size);
		explicit Any(const uint32_t* data, const uint32_t size);
		explicit Any(const float32_t* data, const uint32_t size);
	public:
		void fill_zero();
		bool as_bool() const;
		int32_t as_int32() const;
		uint32_t as_uint32() const;
		float32_t as_float32() const;
		//no copy , no need free by user.
		int32_t* as_int32_array() const;
		uint32_t* as_uint32_array() const;
		float32_t* as_float32_array() const;
	private:
		std::vector<uint8_t> holder;		
	};
}