#include <iostream>
#include <cassert>
#include <algorithm>
#include "../common/mnist_data_loader.h"
#include "../common/utils.h"
#include "easytf/easytf.h"

using namespace easytf;

static EasyTFGraph build_graph(const std::string model_path, const std::string& input_name, const std::string& output_name)
{
	const int32_t batch = 1;
	const int32_t input_size = 784;
	const int32_t fc1_size = 512;
	const int32_t fc2_size = 512;
	const int32_t fc3_size = 10;

	//////////////////////////////////////////////////////////////////////////
	//operators
	//full connect
	const std::vector<float> fc1_w = get_float_array_from_file(model_path, 5);
	easyAssert(fc1_w.size() == input_size*fc1_size, "size of fc1_w is wrong.");
	const std::vector<float> fc1_bias = get_float_array_from_file(model_path, 7);
	easyAssert(fc1_bias.size() == fc1_size, "size of fc1_bias is wrong.");
	auto op_fc1(std::make_shared<OP_FullConnect>(input_size, fc1_size, &fc1_w[0], &fc1_bias[0]));
	auto op_relu1(std::make_shared<OP_Relu>());

	//full connect
	const std::vector<float> fc2_w = get_float_array_from_file(model_path, 14);
	easyAssert(fc2_w.size() == fc1_size*fc2_size, "size of fc2_w is wrong.");
	const std::vector<float> fc2_bias = get_float_array_from_file(model_path, 16);
	easyAssert(fc2_bias.size() == fc2_size, "size of fc2_bias is wrong.");
	auto op_fc2(std::make_shared<OP_FullConnect>(fc1_size, fc2_size, &fc2_w[0], &fc2_bias[0]));
	auto op_relu2(std::make_shared<OP_Relu>());

	//full connect
	const std::vector<float> fc3_w = get_float_array_from_file(model_path, 23);
	easyAssert(fc3_w.size() == fc2_size*fc3_size, "size of fc3_w is wrong.");
	const std::vector<float> fc3_bias = get_float_array_from_file(model_path, 25);
	easyAssert(fc3_bias.size() == fc3_size, "size of fc3_bias is wrong.");
	auto op_fc3(std::make_shared<OP_FullConnect>(fc2_size, fc3_size, &fc3_w[0], &fc3_bias[0]));

	//softmax
	auto op_softmax(std::make_shared<OP_Softmax>());

	//////////////////////////////////////////////////////////////////////////
	//entitys
	std::pair<std::string, std::shared_ptr<Entity>> input{ input_name,
		std::make_shared<Entity>(Shape({ batch, input_size })) };

	//relu & sigmoid & tanh is inplace processed
	std::pair<std::string, std::shared_ptr<Entity>> fc1_output{ "fc1_output",
		std::make_shared<Entity>(Shape({ batch, fc1_size })) };

	std::pair<std::string, std::shared_ptr<Entity>> fc2_output{ "fc2_output",
		std::make_shared<Entity>(Shape({ batch, fc2_size })) };

	std::pair<std::string, std::shared_ptr<Entity>> fc3_output{ "fc3_output",
		std::make_shared<Entity>(Shape({ batch, fc3_size })) };

	std::pair<std::string, std::shared_ptr<Entity>> softmax_output{ output_name,
		std::make_shared<Entity>(Shape({ batch, fc3_size })) };


	EasyTFGraph graph;

	Node node;
	node.op = { "fc1", op_fc1 };
	node.src_entitys.clear();
	node.src_entitys.insert(input);
	node.dst_entitys.clear();
	node.dst_entitys.insert(fc1_output);
	graph.push_node(node);

	node.op = { "relu1", op_relu1 };
	node.src_entitys.clear();
	node.src_entitys.insert(fc1_output);
	node.dst_entitys.clear();
	node.dst_entitys.insert(fc1_output);
	graph.push_node(node);

	node.op = { "fc2", op_fc2 };
	node.src_entitys.clear();
	node.src_entitys.insert(fc1_output);
	node.dst_entitys.clear();
	node.dst_entitys.insert(fc2_output);
	graph.push_node(node);

	node.op = { "relu2", op_relu2 };
	node.src_entitys.clear();
	node.src_entitys.insert(fc2_output);
	node.dst_entitys.clear();
	node.dst_entitys.insert(fc2_output);
	graph.push_node(node);

	node.op = { "fc3", op_fc3 };
	node.src_entitys.clear();
	node.src_entitys.insert(fc2_output);
	node.dst_entitys.clear();
	node.dst_entitys.insert(fc3_output);
	graph.push_node(node);

	node.op = { "softmax", op_softmax };
	node.src_entitys.clear();
	node.src_entitys.insert(fc3_output);
	node.dst_entitys.clear();
	node.dst_entitys.insert(softmax_output);
	graph.push_node(node);

	graph.build();

	return graph;
}

static float test(const std::vector<image_t>& test_images, const std::vector<label_t>& test_labels, const size_t batch = 1)
{
	assert(test_images.size() == test_labels.size() && test_images.size()>0);
	int correctCount = 0;
	for (size_t i = 0; i < test_labels.size(); i += batch)
	{
		const size_t start = i;
		const size_t len = std::min(test_labels.size() - start, batch);
		//do batch test
	}
	const float result = (float)correctCount / (float)test_labels.size();
	return result;
}
static std::vector<float> convert_to_float_array(const std::vector<uint8_t>& data)
{
	std::vector<float> result(data.size());
	for (size_t i = 0; i < data.size();i++)
	{
		result[i] = (float)(data[i]) / 256.0f;
	}
	return result;
}
static void test(const std::string& mnist_test_images_file,
	const std::string& mnist_test_labels_file,
	const std::string& modelFilePath)
{
	bool success = false;

	//load train images
	std::cout << "loading mnist dataset..." << std::endl;
	std::vector<image_t> images;
	success = load_mnist_images(mnist_test_images_file, images);
	assert(success && images.size() > 0);
	//load train labels
	std::vector<label_t> labels;
	success = load_mnist_labels(mnist_test_labels_file, labels);
	assert(success && labels.size() > 0);
	assert(images.size() == labels.size());
	std::cout << "load mnist dataset done." << std::endl;

	//load model and do test
	std::cout << "loading model..." << std::endl;
	const std::string input_name = "input";
	const std::string output_name = "output";
	auto net = build_graph(modelFilePath, input_name, output_name);
	float* input_data = net.get_entity(input_name)->get_data().as_float32_array();
	float* output_data = net.get_entity(output_name)->get_data().as_float32_array();
	const int32_t output_size = net.get_entity(output_name)->get_shape().get_full_size();
	std::cout << "load model done." << std::endl;

	for (size_t i = 0; i < images.size(); i++)
	{
		//fetch data and feed into network
		const std::string label = convert<char,std::string>(labels[i].data+'0');
		std::vector<float> features = convert_to_float_array(images[i].data);
		assert(images[i].data.size() == 784);
		memcpy(input_data, &features[0], sizeof(float)*features.size());
		//run
		net.run();
		//get output from network
		std::pair<int32_t, float> max_item = get_max(output_data, output_size);
		const std::string test_label = convert<char, std::string>(max_item.first+'0');
		const float precision = max_item.second;

		std::cout << "ground truth : " << label << std::endl;
		std::cout << "test result : " << test_label << " (" << precision << ")" << std::endl;
	}
}
int mnist_mlp_main(int argc, char* argv[])
{
	const std::string mnist_test_images_file = "../../resource/mnist_data/t10k-images.idx3-ubyte";
	const std::string mnist_test_labels_file = "../../resource/mnist_data/t10k-labels.idx1-ubyte";
	const std::string mnist_mlp_model = "../../resource/mnist_mlp/mnist_mlp.weights";
	test(mnist_test_images_file, mnist_test_labels_file, mnist_mlp_model);

	return 0;
}