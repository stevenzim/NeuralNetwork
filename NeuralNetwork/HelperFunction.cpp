#include "HelperFunction.h"
#include <iostream>
#include <iomanip>
#include <random>

using namespace std;

/***********************************
Helper functions reusable in multiple applications
************************************/


/***********************************
Sigmoid activation function
************************************/
double HelperFunction::sigmoidFunction(double value , double lambda)
{
	return 1/(1 + (exp (- lambda * value)));
}

/************************************
Create a random weight in the specified range provided
************************************/
double HelperFunction::createRandomWeight(double randMin, double randMax)
{
	std::random_device random;
	double randValue = random() ;  //range 0 to random.max
	randValue = randValue / random.max();  //normalize random number
	double randomRange = randMax - randMin;  //specified user range
	double randomWeight = (randValue * randomRange) - (randomRange / 2.0);  //Calculate random number and shift into correct range
	return randomWeight;
}
