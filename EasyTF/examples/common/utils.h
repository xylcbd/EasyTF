#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>

template<typename SRCT, typename DSTT>
static DSTT convert(const SRCT& content)
{
	std::stringstream ss;
	ss << content;
	DSTT result;
	ss >> result;
	return result;
}
std::vector<std::string> split(const std::string str, const char delimiter);
std::vector<float> convert_to_float_array(const std::string& content, const char split_ch = ',');
std::string& strip(std::string& content);
std::vector<float> get_float_array_from_file(const std::string& file_path, const int line_idx);
std::pair<int32_t, float> get_max(const float* data, const int32_t count);