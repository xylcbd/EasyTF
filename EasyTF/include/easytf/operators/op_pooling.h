#pragma once
#include "easytf/base.h"
#include "easytf/operator.h"

namespace easytf
{
	class OP_Pooling : public Operator
	{
	public:
		enum PoolingType
		{
			MaxPooling,
			MeanPooling
		};
	public:
		//meta string
		//shape
		const static std::string meta_type;
		const static std::string meta_width;
		const static std::string meta_height;
		const static std::string meta_stride_width;
		const static std::string meta_stride_height;
	public:
		OP_Pooling() = default;
		OP_Pooling(const PoolingType type,const int32_t width, const int32_t height);
		//init param
		virtual void init(const Param& param) override;
		//forward
		virtual void forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top) override;
	private:
		//None
		void naive_implement(const int32_t channel,const int32_t src_width,const int32_t src_height,
			const PoolingType type, const int32_t pooling_width, const int32_t pooling_height,
			const float32_t* src,float32_t* dst);
	private:
		//type
		PoolingType type;
		//size
		int32_t width;
		int32_t height;
	};
}