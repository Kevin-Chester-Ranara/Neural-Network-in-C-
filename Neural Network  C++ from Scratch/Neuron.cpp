#include "Neuron.h"

Neuron::Neuron(double val)
	:
	value(val)
{
	Activate();
	DerivedActivated();
}

Neuron::Neuron(double val, int activate_function)
	:
	value(val),
	activation_func(activate_function)
{
	Activate();
	DerivedActivated();
}


void Neuron::Activate()
{
	if(activation_func==RELU)
	{
		//relu
		activatedVal = value > 0 ? value : 0;
	}
	else if (activation_func == SIGMOID)
	{
		//using sigmoid function
		activatedVal = 1 / (1 + exp(-value));
	}
	else if (activation_func == TANH)
	{
		//using tanh function
		activatedVal = tanh(value);
	}

}

void Neuron::DerivedActivated()
{
	if (activation_func == RELU)
	{
		//relu
		Derived_activatedVal = activatedVal > 0 ? 1 : 0;
	}
	else if (activation_func == SIGMOID)
	{
		//sigmoid function
		Derived_activatedVal = activatedVal * (1 - activatedVal);
	}
	else if (activation_func == TANH)
	{
		//tanh
		Derived_activatedVal = (1.0 - (pow(activatedVal, 2)));
	}
	

}

void Neuron::SetValue(double val)
{
	value = val;
	Activate();
	DerivedActivated();
}

double Neuron::GetValue()
{
	return value;
}

double Neuron::GetActivatedVals()
{
	return activatedVal;
}

double Neuron::GetDerivedActivatedVals()
{
	return Derived_activatedVal;
}

void Neuron::Randomize()
{
	std::random_device rd;
	std::mt19937 random(rd());
	std::uniform_real_distribution<double> Dist(0, 1);
	value = Dist(random);
}
