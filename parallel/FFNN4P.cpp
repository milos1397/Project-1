#include "FFNN4P.hpp"

FFNN4P::FFNN4P(unsigned int neuronsInput,unsigned int neuronsHidden1,
			unsigned int neuronsHidden2,unsigned int neuronsOutput, 
			ActivationFunctionTypeP act):
    inputLayer(neuronsInput, neuronsInput, INPUTP, act),
    hiddenLayer1(neuronsInput, neuronsHidden1, HIDDENP, act),
    hiddenLayer2(neuronsHidden1, neuronsHidden2, HIDDENP, act),
    outputLayer(neuronsHidden2, neuronsOutput, OUTPUTP, act),
    input(neuronsInput),
    output(neuronsOutput)
    {
    	activation = act;
    }
    
void FFNN4P::setInput(const std::vector<double>& in)
{   
    if(in.size() != input.size())
        return;
    else
        input = in;
}

void FFNN4P::setWeightMatrices(const std::vector< std::vector <double> >& w1, 
							   const std::vector< std::vector<double> >& w2, 
							   const std::vector< std::vector<double> >& w3)
{
    // Check dimensions
    if(w1.size() != hiddenLayer1.getNeuronsNumLayer() ||
        w1[0].size() != hiddenLayer1.getNeuronsNumPrevLayer() ||
        w2.size() != hiddenLayer2.getNeuronsNumLayer() ||
        w2[0].size() != hiddenLayer2.getNeuronsNumPrevLayer() ||
        w3.size() != outputLayer.getNeuronsNumLayer() ||
        w3[0].size() != outputLayer.getNeuronsNumPrevLayer())
    {
            return;
    }
    else
    {
        hiddenLayer1.setWeightMatrix(w1);
        hiddenLayer2.setWeightMatrix(w2);
        outputLayer.setWeightMatrix(w3);
    }
}

void FFNN4P::action(const std::vector<double>& in, std::vector<double>& out)
{
	std::vector<double> result1 = inputLayer.calculateOutput(in);
    std::vector<double> result2 = hiddenLayer1.calculateOutput(result1);
    std::vector<double> result3 = hiddenLayer2.calculateOutput(result2);
    out = outputLayer.calculateOutput(result3);
}

void FFNN4P::calculateOutput(void)
{
    action(input, output);
}

std::vector<double> FFNN4P::calculateOutput(const std::vector<double>& in)
{
    if(in.size() != inputLayer.getNeuronsNumLayer())
        return std::vector<double>{0};
   	
   	std::vector<double> out(outputLayer.getNeuronsNumLayer());
   	action(in, out);
   	return out;
}

std::vector<double> FFNN4P::getOutput(void) const
{
    return output;
}
