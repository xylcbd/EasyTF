#include <cassert>
#include "easytf/easytf_logger.h"
#include "easytf/easytf_assert.h"

#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif

static void* globalAssertFatalUserData = nullptr;
static void (*globalAssertFatalCB)(void* userData,const std::string& errorStr) = nullptr;

void easytf::setAssertFatalCallback(void(*cb)(void* userData, const std::string& errorStr), void* userData)
{
	globalAssertFatalCB = cb;
	globalAssertFatalUserData = userData;
}
static std::string formatString(const char* fmt, ...) 
{
	std::string s;
	va_list ap;
	va_start(ap, fmt);
	int size = vsnprintf(NULL, 0, fmt, ap);
	va_end(ap);
	if (size > 0) {
		s.resize(size);
		va_start(ap, fmt);
		// Writes the trailing '\0' as well, but we don't care.
		vsprintf(const_cast<char*>(s.data()), fmt, ap);
		va_end(ap);
	}
	return s;
}
void easytf::assertCore(const std::string& file, const std::string& function, const long line,
	const bool condition, const char* fmt, ...)
{
	if (!condition)
	{
		va_list args;
		va_start(args, fmt);
		const std::string stateStr = formatString(fmt, args);
		va_end(args);

		const std::string formatStr = stateStr + "\n  FILE: %s\n  FUNC: %s\n  LINE: %d";
		const std::string messageStr = formatString(formatStr.c_str(), file.c_str(), function.c_str(), line);
		logFatal(messageStr.c_str());
		
		if (globalAssertFatalCB)
		{
			globalAssertFatalCB(globalAssertFatalUserData, messageStr);
		}
		else
		{
			exit(0);
		}
	}
}
