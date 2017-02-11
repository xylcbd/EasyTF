#include "easytf/operators/op_concat.h"
#include "easytf/easytf_assert.h"
#include "easytf/easytf_logger.h"
//meta string
//None

//init
void easytf::OP_Concat::init(const Param& param)
{
}
//forward
void easytf::OP_Concat::forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top)
{
	easyAssert(bottom.size() >= 1 && top.size() == 1, "size of bottom must equals or larger than 1, and top must be 1.");
	auto top_iter = top.begin();
	//check
	const int32_t top_size = top_iter->second->get_shape().get_full_size();
	float32_t* top_data = top_iter->second->get_data().as_float32_array();
	easyAssert(top_data && top_size > 0, "top_data can't be empty.");
	
	std::vector<const float32_t*> src_data;
	std::vector<int32_t> src_size;
	int32_t total_size = 0;
	for (auto iter = bottom.begin(); iter != bottom.end();iter++)
	{
		const int32_t bottom_size = iter->second->get_shape().get_full_size();
		const float32_t* bottom_data = iter->second->get_data().as_float32_array();		
		easyAssert(bottom_data && bottom_size > 0, "bottom_data can't be empty.");
		src_data.push_back(bottom_data);
		src_size.push_back(bottom_size);
		total_size += bottom_size;
	}
	easyAssert(total_size == top_size, "total_size must be equals with top_size.");
	naive_implement(src_data, src_size, top_data, top_size);
}
void easytf::OP_Concat::naive_implement(const std::vector<const float32_t*> src, const std::vector<int32_t>& src_size, float32_t* dst, const int32_t dst_size)
{
	easyAssert(src.size() == src_size.size(), "src.size() must be equals with src_size.size().");
	int32_t offset = 0;
	for (size_t i = 0; i < src.size();i++)
	{
		const int32_t bottom_size = src_size[i];
		const float32_t* bottom_data = src[i];
		memcpy(dst + offset, bottom_data, bottom_size*sizeof(float32_t));
		offset += bottom_size;
	}	
}
