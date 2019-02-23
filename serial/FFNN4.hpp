#ifndef FFNN4_HPP
#define FFNN4_HPP

#include <vector>
#include "Layer.hpp"

/*
Class that models 4-layer Feed-Forward neural network.
*/
class FFNN4
{
    Layer inputLayer;
    Layer hiddenLayer1;
    Layer hiddenLayer2;
    Layer outputLayer;
    std::vector<double> input;
    std::vector<double> output;
    ActivationFunctionType activation;
    void action(const std::vector<double>& in, std::vector<double>& out);
public:
    FFNN4(unsigned int neuronsInput,unsigned int neuronsHidden1,
    	  unsigned int neuronsHidden2,unsigned int neuronsOutput,
    	  ActivationFunctionType act = SIGMOID);
    
    void setInput(const std::vector<double>& in);
    /*
	It's assumed that matrices are regular,
	that means that all columns have same number of elements.
	*/
    void setWeightMatrices(const std::vector< std::vector <double> >& w1, 
    					   const std::vector< std::vector<double> >& w2, 
    					   const std::vector< std::vector<double> >& w3);
    
    // Get inputs from input field and store result in output field.
    void calculateOutput(void);
    
    // Get inputs from in parameter and return results.
    std::vector<double> calculateOutput(const std::vector<double>& in);
    
    std::vector<double> getOutput(void) const;  
};

#endif // FFNN4_HPP
