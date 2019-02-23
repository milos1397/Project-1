#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include "Activation.hpp"

typedef enum {INPUT , HIDDEN, OUTPUT}LayerType;
typedef enum {SIGMOID, IDENTITY, TANH, ARCTAN, BINARY}ActivationFunctionType;

/*
Class that models one layer in Neural Network.
Layer has input, weight matrix, activation function and output.
*/
class Layer
{
	unsigned int neuronsInPreviousLayer;
	unsigned int neuronsInLayer;
    std::vector<double> input;
    std::vector<double> output;
    std::vector< std::vector <double> > weightMatrix;
    LayerType layerType;
    ActivationFunction* activation;
    void action(const std::vector<double>& in, std::vector<double>& out);
public:
    Layer(unsigned int nPrev, unsigned int nLayer, LayerType lType, 
    	  ActivationFunctionType act);
    
    // Copy given input into layers input.
    void setInput(const std::vector<double>& in);
    
    // Copy given weight matrix into layers weight matrix.
    void setWeightMatrix(const std::vector< std::vector<double> >& vM);
    
    std::vector<std::vector<double>> getWeightMatrix(void);

    void calculateOutput(void);
    
    // Calculates and returns Layer's output for given input.
    std::vector<double> calculateOutput(const std::vector<double>& in);
    
    std::vector<double> getOutput(void) const;
    
    unsigned int getNeuronsNumLayer() const;
    unsigned int getNeuronsNumPrevLayer() const;

    // Release memory used by activation function pointer.
    ~Layer();
};

#endif // LAYER_HPP
