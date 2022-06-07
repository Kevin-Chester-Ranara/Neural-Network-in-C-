#pragma once
#include <math.h>
#include <algorithm>
#define RELU 1
#define SIGMOID 2
#define TANH 3

#include <minmax.h>
#include <random>

class Neuron
{
public:
	Neuron(double val);
	Neuron(double val, int activation_func);
	void Activate();
	void DerivedActivated();
	void SetValue(double val);
	double GetValue();
	double GetActivatedVals();
	double GetDerivedActivatedVals();
	void Randomize();
private:
	double value = 0.00;
	double activatedVal;
	double Derived_activatedVal;
	int activation_func = 1;

};