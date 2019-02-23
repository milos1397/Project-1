#include <cmath>
#include "Layer.hpp"

Layer::Layer(unsigned int nPrev, unsigned int nLayer, LayerType lType,  
			 ActivationFunctionType act):
	neuronsInPreviousLayer(nPrev),
	neuronsInLayer(nLayer),
    input(neuronsInPreviousLayer),
   	output(neuronsInLayer),
   	weightMatrix(neuronsInLayer, std::vector<double>(neuronsInPreviousLayer)),
   	layerType(lType)
{	
		if(lType == INPUT)
		{
			neuronsInPreviousLayer = neuronsInLayer;
			input.resize(neuronsInLayer);
		}
		switch(act)
		{
			case SIGMOID:
				activation = new Sigmoid();
				break;
			case IDENTITY:
				activation = new Identity();
				break;
			case TANH:
				activation = new TanH();
				break;
			case ARCTAN:
				activation = new ArcTan();
				break;
			case BINARY:
				activation = new BinaryStep();
				break;
			default:
				activation = new Sigmoid();
				break;
		}
}

void Layer::setInput(const std::vector<double>& in)
{
	if(in.size() != neuronsInPreviousLayer)
		return;
	else
		input = in;
}
    

void Layer::setWeightMatrix(const std::vector< std::vector<double> >& vM)
{
	// If dimensions don't agree return.
	if(vM.size() != neuronsInLayer)
		return;
	for(unsigned int i = 0; i < neuronsInLayer; ++i)
	{
		weightMatrix[i] = vM[i];
	}
}

std::vector<std::vector<double>> Layer::getWeightMatrix(void)
{
	return weightMatrix;
}

void Layer::action(const std::vector<double>& in, std::vector<double>& out)
{
	if(layerType == INPUT)
	{
		// In case of input layer, just apply activation function 
		// to input vector.
		for(unsigned int i = 0; i < neuronsInLayer; ++i)
		{
		    out[i] = activation->calculate(in[i]);
		}
	}
	else
	{
		// Multiply weigth matrix with input vector.
		for(unsigned int i = 0; i < neuronsInLayer; ++i)
		{
		    out[i] = 0;
		    for(unsigned int j = 0; j < neuronsInPreviousLayer; ++j)
		    {
		        out[i] += weightMatrix[i][j] * in[j];
		    }
		    // Apply activation function.
		    out[i] = activation->calculate(out[i]);
		}
	}
}

void Layer::calculateOutput(void)
{
	action(input, output);
}

std::vector<double> Layer::calculateOutput(const std::vector<double>& in)
{

	if(in.size() != neuronsInPreviousLayer)
		return std::vector<double>{0};

	// Temporary output.
	std::vector<double> out(neuronsInLayer);
	action(in ,out);
	return out;
}

std::vector<double> Layer::getOutput(void) const
{
	return output;
}

unsigned int Layer::getNeuronsNumLayer() const
{
	return neuronsInLayer;
}

unsigned int Layer::getNeuronsNumPrevLayer() const
{
	return neuronsInPreviousLayer;
}

Layer::~Layer()
{
	delete activation;
}
