#include "Neural_Network.h"

Neural_Network::Neural_Network(std::vector<int> topo)
	:
	topology(topo)
{
	for (int i = 0; i < topology.size() - 1; i++)
	{
		weightconnections.emplace_back(topology.at(i), topology.at(i + 1), true);
	}
	for (int i = 0; i < topology.at((topology.size()) - 1); i++)
	{
		errors.emplace_back(0.00);
		DerivedErrors.emplace_back(0.00);
	}
	//layers
	for (int i = 0; i < topology.size(); i++)
	{
		if (i == 0)
		{
			//Input Layers
			layers.emplace_back(topology.at(i));
		}
		else if (i > 0 && i < topology.size() - 1)
		{
			//Hidden Layers
			layers.emplace_back(topology.at(i), RELU);
		}
		else if (i == topology.size() - 1)
		{
			//Output Layer
			layers.emplace_back(topology.at(topology.size() - 1), SIGMOID);
		}
	}
}

void Neural_Network::SetTarget(std::vector<double> target)
{
	assert(target.size() == (topology.at(topology.size() - 1))); //sanity check
	targets = target;
}

void Neural_Network::SetInput(std::vector<double> input)
{
	assert(input.size() == topology.at(0));
	layers.at(0).Initialize(input);
}

void Neural_Network::FeedForward()
{
	for (int i = 0; i < topology.size()-1; i++)
	{
		if (i == 0)
		{
			int nextlayer = i + 1;
			Matrix FirstHiddenLayer = (layers.at(i).MatrixValues() * weightconnections.at(i)) + bias;
			layers.at(nextlayer).Initialize(FirstHiddenLayer.Vectorize());
		}
		else
		{
			int nextlayer = i + 1;
			Matrix NextHiddenLayer = (layers.at(i).MatrixActivatedVals() * weightconnections.at(i)) + bias;
			layers.at(nextlayer).Initialize(NextHiddenLayer.Vectorize());
			//layers.at(i).MatrixActivatedVals().Print();
		}
	}
	totalerror = 0.00;
}

void Neural_Network::BackPropagation()
{
	/*
	* OUTPUT TO LAST HIDDEN LAYER
	*/
	//get gradient in the output layer.
	//g=(dC/da)*f'(x); where f'(x) is the derived activation function (sigmoid).
	//[gO1 gO2 gO3]=[y1'*e1'	y2'*e2'		y3'*e3'];
	int OutputLayer = topology.size() - 1;
	assert(topology.at(OutputLayer) == DerivedErrors.size()); //sanity check
	Matrix DerivativeofOutputLayer = layers.at(OutputLayer).MatrixDerivedActivatedVals();
	Matrix DerivativeOfError = Matrix(1, topology.at(OutputLayer), false);
	for (int i = 0; i < DerivedErrors.size(); i++)
	{
		DerivativeOfError.SetValue(0, i, DerivedErrors.at(i));
	}
	Matrix Gradient = Matrix::Hadamard(DerivativeofOutputLayer, DerivativeOfError);
	//deltaweights=(Gradient.transpose * LastHiddenLayer).transpose
	Matrix LasthiddenLayer = layers.at(OutputLayer - 1).MatrixActivatedVals();
	Matrix deltaweights = (Gradient.transpose() * LasthiddenLayer).transpose() * learning_rate;
	Matrix originalweights = weightconnections.at(OutputLayer - 1) * momentum;
	NewWeightConnection.emplace_back(originalweights - deltaweights);
	/*
	* LAST HIDDEN LAYER TO FIRST HIDDEN LAYER
	*/
	for (int i = OutputLayer - 1; i > 0; i--)
	{
		Matrix Gradient_prev = Gradient;
		Matrix WeightConnection_prev = weightconnections.at(i).transpose();
		Matrix Layer_current = layers.at(i).MatrixDerivedActivatedVals();
		Matrix m = Gradient_prev * WeightConnection_prev;
		Gradient = Matrix::Hadamard(m, Layer_current);
		Matrix activatedvaluestranspose = i == 1 ? layers.at(0).MatrixValues().transpose() :
			layers.at(i - 1).MatrixActivatedVals().transpose();
		deltaweights = (activatedvaluestranspose * Gradient) * learning_rate;
		originalweights = weightconnections.at(i - 1) * momentum;
		NewWeightConnection.emplace_back(originalweights - deltaweights);
	}
	/*
	* Updating Weight Connections
	*/
	std::reverse(NewWeightConnection.begin(), NewWeightConnection.end());
	weightconnections.clear();
	weightconnections = NewWeightConnection;
	NewWeightConnection.clear();

}

void Neural_Network::SetErrors()
{
	int outputlayerneurons = topology.at(topology.size() - 1);
	assert(targets.size() == outputlayerneurons);
	switch (CostFunction)
	{
	case MEANSQUAREDERROR:
	{
		std::vector<double> predictedvals = layers.at(topology.size() - 1).GetActivatedVals();
		for (int i = 0; i < targets.size(); i++)
		{
			assert(targets.size() == errors.size());
			double target = targets.at(i);
			double predicted = predictedvals.at(i);
			errors.at(i) = 0.5 * pow((target - predicted), 2);
			DerivedErrors.at(i) = (predicted - target);
			totalerror += errors.at(i);
		}
		break;
	}
	case CROSSENTROPY:

		break;
	default:
		break;
	}
}

void Neural_Network::PrintErrors()
{
	/*for (int i = 0; i < errors.size(); i++)
	{
		std::cout << errors.at(i) << " ";
	}
	std::cout << std::endl;*/
	std::cout << totalerror << std::endl;
}

double Neural_Network::GetTotalErrors()
{
	return totalerror;
}
