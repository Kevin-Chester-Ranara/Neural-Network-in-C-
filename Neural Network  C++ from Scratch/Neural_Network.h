#pragma once
#define MEANSQUAREDERROR 1
#define CROSSENTROPY 2
#include "Layer.h"
#include <algorithm>
#include <fstream>
#include <sstream>
class Neural_Network
{
public:
	Neural_Network(std::vector<int> topology);
	void SetTarget(std::vector<double> target);
	void SetInput(std::vector<double> input);
	void FeedForward();
	void BackPropagation();
	void SetErrors();
	void PrintErrors();
	double GetTotalErrors();
private:
	double learning_rate = 0.01;
	double totalerror = 1;
	double bias = 1;
	double momentum = 1;
	int HiddenActivationType = RELU;
	int OutputActivationType = SIGMOID;
	int CostFunction = MEANSQUAREDERROR;
	std::vector<int> topology;
	std::vector<double> targets;
	std::vector<double> errors;
	std::vector<double> DerivedErrors;
	std::vector<Layer> layers;
	std::vector<Layer> biases;
	std::vector<Matrix> gradientMatrix;
	std::vector<Matrix> NewWeightConnection;
	std::vector<Matrix> weightconnections;
};