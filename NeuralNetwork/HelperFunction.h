#ifndef HELPER_FUNCTION
#define HELPER_FUNCTION

class HelperFunction 
{

public:
	double createRandomWeight (double randMin = -1.0, double randMax = 1.0);
	double sigmoidFunction(double value, double lambda = 1.0);
};

#endif