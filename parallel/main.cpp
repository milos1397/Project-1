#include <iostream>
#include <cstdlib>
#include "FFNN4P.hpp"

// Scales pseudo-random number into [-1,1] range.
#define	RANDOM	(-1.0 +	2.0*(double)rand() / RAND_MAX)

using namespace std;

/*
Initial assumption is that matrix is regular,
that means that all columns has same number of elements.
*/
void RandInitMatrix(vector< vector<double> >& matrix)
{
    const unsigned int rowNum = matrix.size();
    const unsigned int colNum = matrix[0].size();
    for(unsigned int i = 0; i < rowNum; i++)
    {
        for(unsigned int j = 0; j < colNum; j++)
        {
            matrix[i][j] = RANDOM;
        }
    }
}

void RandInitVector(vector<double>& vec)
{
    const unsigned int vecSize = vec.size();
    for(unsigned int i = 0; i < vecSize; i++)
    {
        vec[i] = RANDOM;
    }
}

int main(void)
{   
    // Nerons per layer.
    vector<unsigned int> neurons = {1000, 1000, 1000, 1000};
    // Weight matrices.
    vector<vector<double> > w1(neurons[1], vector<double>(neurons[0]));
    vector< vector<double> > w2(neurons[2], vector<double>(neurons[1]));
    vector< vector<double> > w3(neurons[3], vector<double>(neurons[2]));
    // Input vector.
    vector<double> input(neurons[0]);
    
    // Initialize Matrices.
    RandInitMatrix(w1);
    RandInitMatrix(w2);
    RandInitMatrix(w3);
    // Init vector.   
    RandInitVector(input);
   	
    // Create Network.
    FFNN4P netP(neurons[0], neurons[1], neurons[2], neurons[3]);
    netP.setWeightMatrices(w1, w2, w3);
    vector<double> output = netP.calculateOutput(input);
    
    return 0;
}
