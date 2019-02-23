#ifndef LAYERP_HPP
#define LAYERP_HPP

#include <vector>
#include "ActivationP.hpp"
#include "tbb/tbb.h"
#include "tbb/parallel_reduce.h"
#include "tbb/task.h"

#define P 100000 // Task job parameter.
#define G 10000 // Parallel reduce parameter.

typedef enum {SIGMOIDP, IDENTITYP, TANHP, ARCTANP, BINARYP}
	ActivationFunctionTypeP;

typedef enum{INPUTP , HIDDENP, OUTPUTP}
	LayerTypeP;

/*
Class that models one layer in Neural Network.
Layer has input, weight matrix, activation function and output.
*/
class LayerP
{
	unsigned int neuronsInPreviousLayer;
	unsigned int neuronsInLayer;
    std::vector<double> input;
    std::vector<double> output;
    std::vector< std::vector <double> > weightMatrix;
    LayerTypeP layerType;
    ActivationFunction* activation;
    void action(const std::vector<double>& in, std::vector<double>& out);
public:
    LayerP(unsigned int nPrev, unsigned int nLayer, 
    	   LayerTypeP lType, ActivationFunctionTypeP act);
    
    // Copy given input into layers input.
    void setInput(const std::vector<double>& in);
    
    // Copy given weight matrix into layers weight matrix.
    void setWeightMatrix(const std::vector< std::vector<double> >& vM);
    
    std::vector<std::vector<double>> getWeightMatrix(void);
	
	// Calculates network output, using local input and stores result in local
	// output. 
    void calculateOutput(void);
    
    // Calculates and returns Layer's output for given input.
    std::vector<double> calculateOutput(const std::vector<double>& in);

    std::vector<double> getOutput(void) const;
    
    unsigned int getNeuronsNumLayer() const;
    unsigned int getNeuronsNumPrevLayer() const;

    // Release memory used by activation function pointer.
    ~LayerP();
};

#endif // LAYERP_HPP
