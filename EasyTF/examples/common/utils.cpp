#include "utils.h"

std::vector<std::string> split(const std::string original, const char separator)
{
	std::vector<std::string> results;
	std::string::const_iterator start = original.begin();
	std::string::const_iterator end = original.end();
	std::string::const_iterator next = std::find(start, end, separator);
	while (next != end) {
		results.push_back(std::string(start, next));
		start = next + 1;
		next = std::find(start, end, separator);
	}
	results.push_back(std::string(start, next));
	return results;
}
std::vector<float> convert_to_float_array(const std::string& content, const char split_ch)
{
	std::vector<float> result;
	size_t start = 0;
	while( start <= content.size())
	{
		auto stop = content.find(split_ch, start);
		if (stop == std::string::npos)
		{
			stop = content.size();
		}
		const std::string sub = content.substr(start, stop - start);
		result.push_back((float)atof(sub.c_str()));
		start = stop+1;
	}
	return result;
}
static void find_and_remove(std::string& content, const char ch)
{
	if (content[content.size() - 1] == ch)
	{
		content = content.substr(0, content.size() - 1);
	}
}
std::string& strip(std::string& content)
{
	if (content.empty())
	{
		return content;
	}
	while (true)
	{
		const size_t before_size = content.size();
		find_and_remove(content, '\r');
		find_and_remove(content, '\n');
		find_and_remove(content, '\t');
		find_and_remove(content, ' ');
		if (before_size == content.size())
		{
			break;
		}
	}
	return content;
}
std::vector<float> get_float_array_from_file(const std::string& file_path, const int line_idx)
{
	std::vector<float> result;
	std::ifstream ifs(file_path);
	if (!ifs)
	{
		std::cout << "can not open " << file_path << std::endl;
		return result;
	}
	std::string line;
	for (int i = 1; i <= line_idx; i++)
	{
		std::getline(ifs, line);
	}
	result = convert_to_float_array(strip(line), ' ');
	return result;
}
std::pair<int32_t, float> get_max(const float* data, const int32_t count)
{
	std::pair<int32_t, float> result;
	result.first = 0;
	result.second = data[result.first];

	for (int32_t i = 0; i < count; i++)
	{
		if (data[i] > result.second)
		{
			result.first = i;
			result.second = data[i];
		}
	}

	return result;
}