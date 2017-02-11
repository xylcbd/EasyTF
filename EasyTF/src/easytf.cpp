#include "easytf/easytf.h"
#include "easytf/easytf_assert.h"
#include "easytf/easytf_logger.h"

easytf::EasyTFGraph::EasyTFGraph()
{
	logCritical("EasyTFGraph construct function begin.");
	logCritical("EasyTFGraph construct function end.");
}

void easytf::EasyTFGraph::push_node(const easytf::Node& node)
{
	logCritical("EasyTFGraph::push_node begin.");
	logCritical("op : %s", node.op.first.c_str());
	ops[node.op.first] = node.op.second;

	logCritical("src_entitys:");
	for (auto iter = node.src_entitys.begin(); iter != node.src_entitys.end(); iter++)
	{
		const std::string entity_name = iter->first;
		const std::shared_ptr<Entity> entity = iter->second;
		entitys[entity_name] = entity;
		logCritical("  %s", entity_name.c_str());
	}
	logCritical("dst_entitys:");
	for (auto iter = node.dst_entitys.begin(); iter != node.dst_entitys.end(); iter++)
	{
		const std::string entity_name = iter->first;
		const std::shared_ptr<Entity> entity = iter->second;
		entitys[entity_name] = entity;
		logCritical("  %s", entity_name.c_str());
	}
	nodes.push_back(node);
	
	logCritical("EasyTFGraph::push_node end.");
}
void easytf::EasyTFGraph::build()
{
	logCritical("EasyTFGraph::build begin.");
	//todo
	logCritical("EasyTFGraph::build end.");
}
void easytf::EasyTFGraph::run()
{
	logCritical("EasyTFGraph::run begin.");
	for (size_t i = 0; i < nodes.size();i++)
	{
		Node& node = nodes[i];
		const std::string op_name = node.op.first;
		std::shared_ptr<Operator> op = node.op.second;
		logCritical("op %s forward begin.", op_name.c_str());
		op->forward(node.src_entitys, node.dst_entitys);
		logCritical("op %s forward end.", op_name.c_str());
	}
	logCritical("EasyTFGraph::run end.");
}
std::shared_ptr<easytf::Entity> easytf::EasyTFGraph::get_entity(const std::string& id)
{
	return entitys[id];
}