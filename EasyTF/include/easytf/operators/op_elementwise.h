#pragma once
#include "easytf/base.h"
#include "easytf/operator.h"

namespace easytf
{
	//entity_rs = op(entity_0,entity_1)
	class OP_Elementwise : public Operator
	{
	public:
		//meta string
		//None
	public:
		//init param
		virtual void init(const Param& param) override;
		//forward
		virtual void forward(const std::map<std::string, std::shared_ptr<Entity>>& bottom, std::map<std::string, std::shared_ptr<Entity>>& top) override;
		//action
		virtual void implement(const float32_t* src1, const float32_t* src2, float32_t* dst, const int32_t size) = 0;
		//action
		virtual void implement(const float32_t* src1, const float32_t val, float32_t* dst, const int32_t size) = 0;
	private:
		//None
	private:
		//None
	};

	class OP_Mul : public OP_Elementwise
	{
		//meta string
		//None
	public:
		virtual void implement(const float32_t* src1, const float32_t* src2, float32_t* dst, const int32_t size) override;
		virtual void implement(const float32_t* src1, const float32_t val, float32_t* dst, const int32_t size) override;
	private:
		void naive_implement(const float32_t* src1, const float32_t* src2, float32_t* dst, const int32_t size);
		void naive_implement(const float32_t* src1, const float32_t val, float32_t* dst, const int32_t size);
	private:
		//None
	};

	class OP_Add : public OP_Elementwise
	{
		//meta string
		//None
	public:
		virtual void implement(const float32_t* src1, const float32_t* src2, float32_t* dst, const int32_t size) override;
		virtual void implement(const float32_t* src1, const float32_t val, float32_t* dst, const int32_t size) override;
	private:
		void naive_implement(const float32_t* src1, const float32_t* src2, float32_t* dst, const int32_t size);
		void naive_implement(const float32_t* src1, const float32_t val, float32_t* dst, const int32_t size);
	private:
		//None
	};
}