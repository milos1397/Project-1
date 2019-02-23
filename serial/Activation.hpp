#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <cmath>

// Base abstract class for all activation functions.
class ActivationFunction
{
public:
	virtual double calculate(double x) = 0;
};

class Sigmoid : public ActivationFunction
{
public:
	double calculate(double x)
	{
		return (1 / (1 + exp(-1.0 * x)));
	}
};

class Identity : public ActivationFunction
{
public:
	double calculate(double x)
	{
		return x;
	}
};

class TanH : public ActivationFunction
{
public:
	double calculate(double x)
	{
		return ((exp(x) - exp(-1.0 * x)) / (exp(x) + exp(-1.0 * x)));
	}
};

class ArcTan : public ActivationFunction
{
public:
	double calculate(double x)
	{
		return atan(-1.0 * x);
	}
};

class BinaryStep : public ActivationFunction
{
public:
	double calculate(double x)
	{
		return (x < 0) ? 0 : 1;
	}
};

#endif // ACTIVATION_HPP
