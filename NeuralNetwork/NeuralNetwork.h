#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <vector>
#include <string>
#include "HelperFunction.h"

class NeuralNetwork
{
private:
	//Create helper function object
	HelperFunction helper;

	//Neural network vars and paramaters
	int numInputs, numHiddens, numOutputs, numInsWithBias, numHiddensWithBias, totalNumWeights;
	double eta, alpha, lambda;

	//INPUT TO HIDDEN LAYER matrices
	std::vector<double> inputs;
	std::vector< std::vector<double> >inputToHiddenWeights;
	std::vector< std::vector<double> >inputToHiddenCurrentDeltaWeights;
	std::vector< std::vector<double> >inputToHiddenPreviousDeltaWeights;
	//INPUT TO HIDDEN LAYER vectors
	std::vector<double> inputToHiddenSums;
	std::vector<double> hiddenPredictions;
	std::vector<double> hiddenGradients;


	//HIDDEN TO OUTPUT LAYER matrices
	std::vector< std::vector<double> >hiddenToOutputWeights;
	std::vector< std::vector<double> >hiddenToOutputCurrentDeltaWeights;
	std::vector< std::vector<double> >hiddenToOutputPreviousDeltaWeights;
	//HIDDEN TO OUTPUT LAYER vectors 
	std::vector<double> hiddenToOutputSums;
	std::vector<double> outputPredictions;
	std::vector<double> outputs;
	std::vector<double> outputGradients;
	std::vector<double> outputErrors; 
	std::vector<double> batchSumErrors; 

	//These are the sum of errors for an entire epoch for all outputs
	double sumMeanEpochOutputError;
	double sumBatchError;


	//MAX-MINS FOR INPUTS AND OUTPUTS
	//Note: input max and mins must have same number of elements (same for output max mins)
	std::vector<double> inputMaxes;
	std::vector<double> inputMins;
	std::vector<double> outputMaxes;
	std::vector<double> outputMins;



public:
	/***********************************
	Neural Network Constructor and initialization
	************************************/
	NeuralNetwork(int numInputs, int numHiddens, int numOutputs, double eta, double alpha, double lambda);


	/***********************************
	set weights to random initial values
	************************************/
	void initializeWeightMatrices (void); // for setting weights to initial random values

	/***********************************
	set normalization paramaters
	************************************/
	void setNormalizationParams(double inMax [], double inMin [], double outMax [], double outMin[]);

	/************************************************************
	Feed Forward Algorithm
	NOTE: This method can take vector with 
	size = number of inputs or total number of inputs and outputs 
	This is done in order to handle the two applications of feedforward
	1) Training, where we want to store the outputs for backpropagation
	2) Operational, where we only want the predicted outputs 
	************************************************************/
	void feedForward(std::vector <double> trainingExample);

	/************************************************************
	Back Propagation Algorithm
	NOTE: This method, given errors calculated in forward propagate,
	updates the weights and delta weights.
	************************************************************/
	void backPropagate();

	/************************************************************
	Calculate the RMSE for the current online epoch (non Batch)
	************************************************************/
	double calcRMSE(int numExamples);

	/************************************************************
	Denormalize the output
	************************************************************/
	void denormOutputPredictions(std::vector<double> &robot);



	/************************************************************
	Remaining code in this class is necessary for batch update process for robot
	************************************************************/

	/************************************************************
	Reset all current and previous delta weights
	NOTE: The overall weights
	************************************************************/
	void resetPreviousDeltaWeights(void);

	/************************************************************
	Set outputs to current values found to be closest to example in training data set
	Also, sum the batch  squared error
	************************************************************/
	double batchLearn(int numExamples);

	/************************************************************
	Runs at the end of each Batch learningepoch,
	1) calculates the RMSE for the entire batch
	2) sets the output Error for each wheel to the RMSE for that wheel
	3) resets batchSumErrors
	************************************************************/
	void batchSetOutputs(std::vector <double> onlineTrainOutputs,std::vector <double> onlineTrainOutputsPreds);
};


#endif