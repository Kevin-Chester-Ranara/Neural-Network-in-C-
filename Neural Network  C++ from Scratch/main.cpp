#include <fstream>
#include <streambuf>
#include <ostream>
#include <iostream>
#include "Neural_Network.h"

int main()
{
	/*Matrix a = Matrix(3, 3, true);
	a.Print();
	Matrix b = Matrix(3, 3, true);
	b.Print();
	Matrix c = a * b + 25;
	c.Print();
	std::vector<double> vec = c.Vectorize();
	for (int i = 0; i < vec.size(); i++)
	{
		std::cout << vec.at(i) << std::endl;
	}
	Matrix d=c.transpose();
	d.Print();*/
	std::vector<int> topo = { 4,6,3,3 };
	std::vector<double> target = { 0.2,0.5,0.1 };
	std::vector<double> input = { 0.2,0.5,0.1,0.6 };
	Neural_Network nn = Neural_Network(topo);
	nn.SetTarget(target);
	nn.SetInput(input);
	while (nn.GetTotalErrors() > 0.000001)
	{
		nn.FeedForward();
		nn.SetErrors();
		nn.PrintErrors();
		nn.BackPropagation();

	}
	

	return 0;
}