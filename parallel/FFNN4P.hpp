#ifndef FFNN4P_HPP
#define FFNN4P_HPP

#include <vector>
#include "LayerP.hpp"

/*
Class that models 4-layer Feed-Forward neural network.
*/
class FFNN4P
{
    LayerP inputLayer;
    LayerP hiddenLayer1;
    LayerP hiddenLayer2;
    LayerP outputLayer;
    std::vector<double> input;
    std::vector<double> output;
    ActivationFunctionTypeP activation;
    void action(const std::vector<double>& in, std::vector<double>& out);
public:
    FFNN4P(unsigned int neuronsInput,unsigned int neuronsHidden1,
    	  unsigned int neuronsHidden2,unsigned int neuronsOutput, 
    	  ActivationFunctionTypeP act = SIGMOIDP);
    
    void setInput(const std::vector<double>& in);
    
    void setWeightMatrices(const std::vector< std::vector <double> >& w1, 
    					   const std::vector< std::vector<double> >& w2,
    					   const std::vector< std::vector<double> >& w3);
    
    // Get inputs from input field and store result in output field.
    void calculateOutput(void);
    
    // Get inputs from in parameter and return results.
    std::vector<double> calculateOutput(const std::vector<double>& in);
    
    std::vector<double> getOutput(void) const;  
};

#endif // FFNN4P_HPP
