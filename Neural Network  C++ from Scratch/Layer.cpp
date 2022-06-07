#include "Layer.h"

Layer::Layer(int num)
    :
    layer_num(num)
{
    double initial = 0.00;
    for (int i = 0; i < layer_num; i++)
    {
        neurons.emplace_back(initial);
    }
}

Layer::Layer(int num, int Activation_function)
    :
    layer_num(num)
{
    double initial = 0.00;
    for (int i = 0; i < layer_num; i++)
    {
        neurons.emplace_back(initial, Activation_function);
    }
}

void Layer::Initialize(std::vector<double> values)
{
    assert(layer_num == values.size());
    for (int i = 0; i < layer_num; i++)
    {
        neurons.at(i).SetValue(values.at(i));
    }
}

void Layer::SetValue(int column, double val)
{
    neurons.at(column).SetValue(val);
}

Matrix Layer::MatrixValues()
{
    Matrix c = Matrix(1, layer_num, false);
    for (int i = 0; i < layer_num; i++)
    {
        c.SetValue(0, i, neurons.at(i).GetValue());
    }
    return c;
}

Matrix Layer::MatrixActivatedVals()
{
    Matrix c = Matrix(1, layer_num, false);
    for (int i = 0; i < layer_num; i++)
    {
        c.SetValue(0, i, neurons.at(i).GetActivatedVals());
    }
    return c;
}

Matrix Layer::MatrixDerivedActivatedVals()
{
    Matrix c = Matrix(1, layer_num, false);
    for (int i = 0; i < layer_num; i++)
    {
        c.SetValue(0, i, neurons.at(i).GetDerivedActivatedVals());
    }
    return c;
}

std::vector<Neuron> Layer::GetNeurons()
{
    return neurons;
}

std::vector<double> Layer::GetActivatedVals()
{
    std::vector<double> activatedvals;
    for (int i = 0; i < layer_num; i++)
    {
        activatedvals.emplace_back(neurons.at(i).GetActivatedVals());
    }
    return activatedvals;
}
