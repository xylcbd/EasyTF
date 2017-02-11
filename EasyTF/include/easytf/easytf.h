#pragma once
//utils
#include "easytf/easytf_assert.h"
#include "easytf/easytf_logger.h"
//entity
#include "easytf/entity.h"
//operators
#include "easytf/operator.h"
#include "easytf/operators/op_lstm.h"
#include "easytf/operators/op_fullconnect.h"
#include "easytf/operators/op_relu.h"
#include "easytf/operators/op_sigmoid.h"
#include "easytf/operators/op_tanh.h"
#include "easytf/operators/op_softmax.h"
#include "easytf/operators/op_concat.h"
#include "easytf/operators/op_reverse.h"

namespace easytf
{
	struct Node
	{
		std::pair<std::string, std::shared_ptr<Operator>> op;
		std::map<std::string, std::shared_ptr<Entity>> src_entitys;
		std::map<std::string, std::shared_ptr<Entity>> dst_entitys;
	};

	class EasyTFGraph
	{
	public:
		EasyTFGraph();
	public:
		//build network
		void push_node(const Node& node);
		void build();
		//before run , you must feed data to input entity!
		void run();
		//get result
		std::shared_ptr<Entity> get_entity(const std::string& id);
	private:
		std::map<std::string, std::shared_ptr<Entity>> entitys;
		std::map<std::string, std::shared_ptr<Operator>> ops;
		std::vector<Node> nodes;
	};
}