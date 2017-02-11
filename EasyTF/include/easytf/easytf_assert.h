#pragma once
#include <string>
#include "easytf/base.h"

namespace easytf
{
	void setAssertFatalCallback(void (*cb)(void* userData,const std::string& errorStr),void* userData);
	void assertCore(const std::string& file,const std::string& function,const long line,
		const bool condition, const char* fmt, ...);
#define easyAssert(condition,fmt,...) \
	assertCore(__FILE__,__FUNCTION__,__LINE__,(condition),(fmt),__VA_ARGS__);
}