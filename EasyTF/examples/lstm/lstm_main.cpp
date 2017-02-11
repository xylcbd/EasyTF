#include "easytf/easytf.h"
#include "../common/utils.h"

using namespace easytf;

#define USING_BILSTM 1

#if USING_BILSTM
static EasyTFGraph build_graph(const std::string model_path, const std::string& input_name, const std::string& output_name)
{
	const int32_t batch = 1;
	const int32_t input_dim = 6;
	const int32_t lstm_max_frames = 100;
	const int32_t lstm_dim = 64;
	const int32_t fc1_size = 48;
	const int32_t fc2_size = 6;
	const bool lstm_sequence = false;

	//////////////////////////////////////////////////////////////////////////
	//operators
	//lstm
	const std::vector<float> lstm1_wih = get_float_array_from_file(model_path, 10);
	easyAssert(lstm1_wih.size() == 4 * input_dim*lstm_dim, "size of lstm1_wih is wrong.");
	const std::vector<float> lstm1_whh = get_float_array_from_file(model_path, 12);
	easyAssert(lstm1_whh.size() == 4 * lstm_dim*lstm_dim, "size of lstm1_whh is wrong.");
	const std::vector<float> lstm1_bias = get_float_array_from_file(model_path, 14);
	easyAssert(lstm1_bias.size() == 4 * lstm_dim, "size of lstm1_bias is wrong.");
	auto op_lstm1(std::make_shared<OP_LSTM>(input_dim, lstm_dim, false, 
		&lstm1_wih[0], &lstm1_whh[0], &lstm1_bias[0]));

	//reverse lstm
	auto op_reverse1(std::make_shared<OP_Reverse>(input_dim));
	const std::vector<float> lstm1_reverse_wih = get_float_array_from_file(model_path, 16);
	easyAssert(lstm1_reverse_wih.size() == 4 * input_dim*lstm_dim, "size of lstm1_reverse_wih is wrong.");
	const std::vector<float> lstm1_reverse_whh = get_float_array_from_file(model_path, 18);
	easyAssert(lstm1_reverse_whh.size() == 4 * lstm_dim*lstm_dim, "size of lstm1_reverse_whh is wrong.");
	const std::vector<float> lstm1_reverse_bias = get_float_array_from_file(model_path, 20);
	easyAssert(lstm1_reverse_bias.size() == 4 * lstm_dim, "size of lstm1_reverse_bias is wrong.");
	auto op_lstm1_reverse(std::make_shared<OP_LSTM>(input_dim, lstm_dim, false,
		&lstm1_reverse_wih[0], &lstm1_reverse_whh[0], &lstm1_reverse_bias[0]));

	//concat
	auto op_add1(std::make_shared<OP_Add>());
	const int32_t add_dim = lstm_dim;

	//full connect
	const std::vector<float> fc1_w = get_float_array_from_file(model_path, 24);
	easyAssert(fc1_w.size() == add_dim*fc1_size, "size of fc1_w is wrong.");
	const std::vector<float> fc1_bias = get_float_array_from_file(model_path, 26);
	easyAssert(fc1_bias.size() == fc1_size, "size of fc1_bias is wrong.");
	auto op_fc1(std::make_shared<OP_FullConnect>(add_dim, fc1_size, &fc1_w[0], &fc1_bias[0]));
	auto op_relu1(std::make_shared<OP_Relu>());

	//full connect
	const std::vector<float> fc2_w = get_float_array_from_file(model_path, 32);
	easyAssert(fc2_w.size() == fc1_size*fc2_size, "size of fc2_w is wrong.");
	const std::vector<float> fc2_bias = get_float_array_from_file(model_path, 34);
	easyAssert(fc2_bias.size() == fc2_size, "size of fc2_bias is wrong.");
	auto op_fc2(std::make_shared<OP_FullConnect>(fc1_size, fc2_size, &fc2_w[0], &fc2_bias[0]));

	//softmax
	auto op_softmax(std::make_shared<OP_Softmax>());

	//////////////////////////////////////////////////////////////////////////
	//entitys
	std::pair<std::string, std::shared_ptr<Entity>> entity_input{ input_name,
		std::make_shared<Entity>(Shape({ batch, lstm_max_frames, input_dim })) };

	std::pair<std::string, std::shared_ptr<Entity>> lstm1_output{ "lstm1_output", 
		std::make_shared<Entity>(Shape({ batch, 1, lstm_dim })) };

	//FIXME : rever frames
	std::pair<std::string, std::shared_ptr<Entity>> reverse1_output{ "reverse1_output",
		std::make_shared<Entity>(entity_input.second->get_shape()) };
	std::pair<std::string, std::shared_ptr<Entity>> lstm1_reverse_output{ "lstm1_reverse_output",
		std::make_shared<Entity>(lstm1_output.second->get_shape()) };

	std::pair<std::string, std::shared_ptr<Entity>> add1_output{ "add1_output",
		std::make_shared<Entity>(Shape({ batch, 1, add_dim })) };

	//relu & sigmoid & tanh is inplace processed
	std::pair<std::string, std::shared_ptr<Entity>> fc1_output{ "fc1_output", 
		std::make_shared<Entity>(Shape({ batch, fc1_size })) };

	std::pair<std::string, std::shared_ptr<Entity>> fc2_output{ "fc2_output", 
		std::make_shared<Entity>(Shape({ batch, fc2_size })) };

	std::pair<std::string, std::shared_ptr<Entity>> softmax_output{ output_name,
		std::make_shared<Entity>(fc2_output.second->get_shape()) };


	EasyTFGraph graph;

	Node node;
	node.op = { "lstm1", op_lstm1 };
	node.src_entitys.clear();
	node.src_entitys.insert(entity_input);
	node.dst_entitys.clear();
	node.dst_entitys.insert(lstm1_output);
	graph.push_node(node);

	node.op = { "reverse1", op_reverse1 };
	node.src_entitys.clear();
	node.src_entitys.insert(entity_input);
	node.dst_entitys.clear();
	node.dst_entitys.insert(reverse1_output);
	graph.push_node(node);

	node.op = { "lstm1_reverse", op_lstm1_reverse };
	node.src_entitys.clear();
	node.src_entitys.insert(reverse1_output);
	node.dst_entitys.clear();
	node.dst_entitys.insert(lstm1_reverse_output);
	graph.push_node(node);

	node.op = { "add1", op_add1 };
	node.src_entitys.clear();
	node.src_entitys.insert(lstm1_output);
	node.src_entitys.insert(lstm1_reverse_output);
	node.dst_entitys.clear();
	node.dst_entitys.insert(add1_output);
	graph.push_node(node);

	node.op = { "fc1", op_fc1 };
	node.src_entitys.clear();
	node.src_entitys.insert(add1_output);
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

	node.op = { "softmax", op_softmax };
	node.src_entitys.clear();
	node.src_entitys.insert(fc2_output);
	node.dst_entitys.clear();
	node.dst_entitys.insert(softmax_output);
	graph.push_node(node);

	graph.build();

	return graph;
}
#else
static EasyTFGraph build_graph(const std::string model_path, const std::string& input_name, const std::string& output_name)
{
	const int32_t batch = 1;
	const int32_t input_dim = 6;
	const int32_t lstm_max_frames = 100;
	const int32_t lstm_dim = 64;
	const int32_t fc1_size = 48;
	const int32_t fc2_size = 6;
	const bool lstm_sequence = false;

	//////////////////////////////////////////////////////////////////////////
	//operators
	//lstm
	const std::vector<float> lstm1_wih = get_float_array_from_file(model_path, 12);
	easyAssert(lstm1_wih.size() == 4 * input_dim*lstm_dim, "size of lstm1_wih is wrong.");
	const std::vector<float> lstm1_whh = get_float_array_from_file(model_path, 14);
	easyAssert(lstm1_whh.size() == 4 * lstm_dim*lstm_dim, "size of lstm1_whh is wrong.");
	const std::vector<float> lstm1_bias = get_float_array_from_file(model_path, 16);
	easyAssert(lstm1_bias.size() == 4 * lstm_dim, "size of lstm1_bias is wrong.");
	auto op_lstm1(std::make_shared<OP_LSTM>(input_dim, lstm_dim, false,
		&lstm1_wih[0], &lstm1_whh[0], &lstm1_bias[0]));

	//full connect
	const std::vector<float> fc1_w = get_float_array_from_file(model_path, 20);
	easyAssert(fc1_w.size() == lstm_dim*fc1_size, "size of fc1_w is wrong.");
	const std::vector<float> fc1_bias = get_float_array_from_file(model_path, 22);
	easyAssert(fc1_bias.size() == fc1_size, "size of fc1_bias is wrong.");
	auto op_fc1(std::make_shared<OP_FullConnect>(lstm_dim, fc1_size, &fc1_w[0], &fc1_bias[0]));
	auto op_relu1(std::make_shared<OP_Relu>());

	//full connect
	const std::vector<float> fc2_w = get_float_array_from_file(model_path, 28);
	easyAssert(fc2_w.size() == fc1_size*fc2_size, "size of fc2_w is wrong.");
	const std::vector<float> fc2_bias = get_float_array_from_file(model_path, 30);
	easyAssert(fc2_bias.size() == fc2_size, "size of fc2_bias is wrong.");
	auto op_fc2(std::make_shared<OP_FullConnect>(fc1_size, fc2_size, &fc2_w[0], &fc2_bias[0]));

	//softmax
	auto op_softmax(std::make_shared<OP_Softmax>());

	//////////////////////////////////////////////////////////////////////////
	//entitys
	std::pair<std::string, std::shared_ptr<Entity>> entity_input{ input_name,
		std::make_shared<Entity>(Shape({ batch, lstm_max_frames, input_dim })) };

	std::pair<std::string, std::shared_ptr<Entity>> lstm1_output{ "lstm1_output",
		std::make_shared<Entity>(Shape({ batch, 1, lstm_dim })) };

	//relu & sigmoid & tanh is inplace processed
	std::pair<std::string, std::shared_ptr<Entity>> fc1_output{ "fc1_output",
		std::make_shared<Entity>(Shape({ batch, fc1_size })) };

	std::pair<std::string, std::shared_ptr<Entity>> fc2_output{ "fc2_output",
		std::make_shared<Entity>(Shape({ batch, fc2_size })) };

	std::pair<std::string, std::shared_ptr<Entity>> softmax_output{ output_name,
		std::make_shared<Entity>(fc2_output.second->get_shape()) };


	EasyTFGraph graph;

	Node node;
	node.op = { "lstm1", op_lstm1 };
	node.src_entitys.clear();
	node.src_entitys.insert(entity_input);
	node.dst_entitys.clear();
	node.dst_entitys.insert(lstm1_output);
	graph.push_node(node);

	node.op = { "fc1", op_fc1 };
	node.src_entitys.clear();
	node.src_entitys.insert(lstm1_output);
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

	node.op = { "softmax", op_softmax };
	node.src_entitys.clear();
	node.src_entitys.insert(fc2_output);
	node.dst_entitys.clear();
	node.dst_entitys.insert(softmax_output);
	graph.push_node(node);

	graph.build();

	return graph;
}
#endif //USING_BILSTM

static std::vector<std::pair<std::string, std::vector<std::vector<float>>>> get_test_data(const std::string& data_path)
{
	std::vector<std::pair<std::string, std::vector<std::vector<float>>>> result;
	std::ifstream ifs(data_path);
	while (true)
	{
		std::string line;
		//label
		std::getline(ifs, line);
		strip(line);
		if (line.empty())
		{
			break;
		}
		const std::string label = line;
		//feature count
		std::getline(ifs, line);
		strip(line);
		const uint16_t count = convert<std::string,uint16_t>(line);
		//feature details
		std::vector<std::vector<float>> features;
		for (uint16_t i = 0; i < count;i++)
		{
			std::getline(ifs, line);
			strip(line);
			const std::vector<float> feature = convert_to_float_array(line);
			features.push_back(feature);
		}
		result.push_back({ label, features });
	}
	return result;
}

static void test_lstm()
{
#if USING_BILSTM
	const std::string model_path = "../../resource/lstm/bilstm.weights";
#else
	const std::string model_path = "../../resource/lstm/lstm.weights";
#endif
	std::cout << "loading test data..." << std::endl;
	const auto test_samples = get_test_data("../../resource/lstm/classify_lstm_test.db");
	std::cout << "load test data done." << std::endl;

	std::cout << "loading model..." << std::endl;
	const std::string& input_name = "net_input";
	const std::string& output_name = "net_output";
	EasyTFGraph graph(build_graph(model_path, input_name, output_name));
	float* input_data = graph.get_entity(input_name)->get_data().as_float32_array();
	float* output_data = graph.get_entity(output_name)->get_data().as_float32_array();
	const int32_t output_size = graph.get_entity(output_name)->get_shape().get_full_size();
	std::cout << "load model done." << std::endl;

	int total = 0;
	int correct = 0;
	for (size_t i = 0; i < test_samples.size(); i++)
	{
		//zero it!
		graph.get_entity(input_name)->get_data().fill_zero();

		//fetch data and feed into network
		const std::string label = test_samples[i].first;
		std::vector<std::vector<float>> features = test_samples[i].second;
		for (size_t j = 0; j < features.size();j++)
		{
			const int feature_dim = features[j].size();
			memcpy(input_data + j*feature_dim, &features[j][0], sizeof(float)*feature_dim);
		}
		//run
		graph.run();
		//get output from network
		std::pair<int32_t, float> max_item = get_max(output_data, output_size);
		const std::string test_label = convert<char,std::string>(max_item.first+'0');
		const float precision = max_item.second;

		std::cout << "ground truth : " << label << std::endl;
		std::cout << "test result : " << test_label << " ("<<precision<<")" << std::endl;	
		if (label == test_label)
		{
			correct++;
		}
		total++;
	}	
	const float accuracy = float(correct) / float(total);
	std::cout << "accuracy : " << accuracy << std::endl;
}
int lstm_main(int argc, char* argv[])
{
	test_lstm();
	return 0;
}
