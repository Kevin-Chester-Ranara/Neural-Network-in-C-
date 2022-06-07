#pragma once
#include <vector>
#include "Neuron.h"
#include "Matrix.h"
class Layer
{
public:
	Layer(int num);
	Layer(int num, int Activation_function);
	void Initialize(std::vector<double> values);
	void SetValue(int column, double val);
	Matrix MatrixValues();
	Matrix MatrixActivatedVals();
	Matrix MatrixDerivedActivatedVals();
	std::vector<Neuron> GetNeurons();
	std::vector<double> GetActivatedVals();
private:
	int layer_num;
	std::vector<Neuron> neurons;
};