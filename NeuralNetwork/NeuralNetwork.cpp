#include "HelperFunction.h"
#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

using namespace std;

/***********************************
Neural Network Constructor and initialization
************************************/
NeuralNetwork::NeuralNetwork(int numInputs, int numHiddens, int numOutputs, double eta, double alpha, double lambda) 
	: numInputs(numInputs), numHiddens(numHiddens), numOutputs (numOutputs), eta(eta), alpha(alpha), lambda(lambda)
{	
	//Additional variables for matrix, array and vector initializations
	numInsWithBias = numInputs + 1;
	numHiddensWithBias = numHiddens + 1;
	totalNumWeights = numInsWithBias * numHiddensWithBias * numOutputs;

	//INPUT TO HIDDEN LAYER  Matrices ( NOTE + 1 is used to include biases)
	inputs.resize(numInputs + 1);
	inputToHiddenWeights.resize((numHiddens) , vector<double>(numInputs + 1));
	inputToHiddenCurrentDeltaWeights.resize((numHiddens) , vector<double>(numInputs + 1));
	inputToHiddenPreviousDeltaWeights.resize((numHiddens) , vector<double>(numInputs + 1));
	//INPUT TO HIDDEN VECTORS
	inputToHiddenSums.resize(numHiddens + 1);
	hiddenPredictions.resize(numHiddens + 1);
	hiddenGradients.resize(numHiddens + 1);

	//HIDDEN TO OUTPUT LAYER Matrices
	hiddenToOutputWeights.resize((numOutputs) , vector<double>(numHiddens + 1));
	hiddenToOutputCurrentDeltaWeights.resize((numOutputs) , vector<double>(numHiddens + 1));
	hiddenToOutputPreviousDeltaWeights.resize((numOutputs) , vector<double>(numHiddens + 1));
	//HIDDEN TO OUTPUT LAYER VECTORS
	hiddenToOutputSums.resize(numOutputs);
	outputPredictions.resize(numOutputs);
	outputs.resize(numOutputs);
	outputGradients.resize(numOutputs);
	outputErrors.resize(numOutputs); 
	batchSumErrors.resize(numOutputs); //this collects and stores the sum of errors for an entire epoch for each output

	//These are the sum of errors for an entire epoch for all outputs
	sumMeanEpochOutputError = 0.0;
	sumBatchError = 0.0;

	//INITIALIZE BIAS UNITS
	inputs[0] = 1.0;
	inputToHiddenSums[0] = 1.0;
	hiddenPredictions[0] = 1.0;

	//MAX-MINS FOR INPUTS AND OUTPUTS
	//Note: input max and mins must have same number of elements (same for output max mins)
	inputMaxes.resize (numInputs);
	inputMins.resize(numInputs);
	outputMaxes.resize(numOutputs);
	outputMins.resize(numOutputs);
}


/***********************************
set weights to random initial values
************************************/
void NeuralNetwork::initializeWeightMatrices ()
{
	//FUTURE WORK: Since these functions and functions in setWeightMatrices are similar, consider way to abstract out.
	// i = unit weight is going into
	// j = unit weight is coming from
	//initialize input to hidden weights
	for (unsigned int i = 0; i < inputToHiddenWeights.size(); i++)
	{
		for (unsigned int j = 0; j < inputToHiddenWeights[i].size(); j++)
		{
			inputToHiddenWeights[i][j] = helper.createRandomWeight();
		}
	};

	//initialize hidden to output weights
	for (unsigned int i = 0; i < hiddenToOutputWeights.size(); i++)
	{
		for (unsigned int j = 0; j < hiddenToOutputWeights[i].size(); j++)
		{
			hiddenToOutputWeights[i][j] = helper.createRandomWeight();
		}
	};
}

/***********************************
set normalization paramaters
************************************/
void NeuralNetwork::setNormalizationParams(double inMax [], double inMin [], double outMax [], double outMin[])
{
	//FUTURE WORK: Consider a better solution for maxes and mins and normalization.
	for (int i = 0; i < numInputs; i++)
	{
		inputMaxes[i] = inMax[i];
		inputMins[i] = inMin[i];
	}

	for (int i = 0; i < numOutputs; i++)
	{
		outputMaxes[i] = outMax[i];
		outputMins[i] = outMin[i];
	}
}

/************************************************************
Feed Forward Algorithm
NOTE: This method can take vector with 
size = number of inputs or total number of inputs and outputs 
This is done in order to handle the two applications of feedforward
1) Training, where we want to store the outputs for backpropagation
2) Operational, where we only want the predicted outputs 
************************************************************/
void NeuralNetwork::feedForward(std::vector <double> trainingExample)
{

	//initialize counter so algorithm knows which element of vector is being read
	int idxExample = 0;

	//FUTURE WORK: Handler to throw exception OR toss out training example if total elements != inputs + outputs

	//Normalise inputs and outputs
	//FUTURE WORK: Consider if normalization of the data prior to loading passing into feedforward is better.  
	//		i.e. normalize the data from file and Don't do it each time in feedforward.
	// i = ith feature of inputs
	for (unsigned int i = 1; i < inputs.size(); i++)
	{
		inputs[i] = ((trainingExample[idxExample] - inputMins[i - 1]) / (inputMaxes[i - 1] - inputMins[i - 1])); 
		idxExample++;
	}

	// i = ith feature of outputs
	if (trainingExample.size() > inputs.size())
	{ 	/*only udate outputs if training, not if only predictions are needed
		i.e. this does not run if you are trying to produce predicted outputs only */
		for (unsigned int i = 0; i < outputs.size(); i++)
		{
			outputs[i] = ((trainingExample[idxExample] - outputMins[i]) / (outputMaxes[i] - outputMins[i])); 
			idxExample++;
		}
	}

	//Calculate hidden neuron weighted sums and predictions
	//i = number of inputs + bias j = number of hiddens
	for (unsigned int i = 0; i < inputToHiddenWeights.size() ; i++)
	{
		double sum = 0.0;
		for (unsigned int j = 0; j < inputToHiddenWeights[i].size() ; j++)
		{
			sum = sum + inputToHiddenWeights[i][j] * inputs[j];
		}
		inputToHiddenSums[i + 1] = sum;
		hiddenPredictions[i + 1] = helper.sigmoidFunction(sum);
	}

	//Calculate output weighted sums and predictions
	//i = number of hiddens + bias j = number of outputs
	for (unsigned int i = 0; i < hiddenToOutputWeights.size() ; i++)
	{
		double sum = 0.0;
		for (unsigned int j = 0; j < hiddenToOutputWeights[i].size() ; j++)
		{
			sum = sum + hiddenToOutputWeights[i][j] * hiddenPredictions[j];
		}
		hiddenToOutputSums[i] = sum ;
		outputPredictions[i] = helper.sigmoidFunction(sum);
	}

	//Calculate outputErrors, update sum errors
	double sumCurrentExampleError = 0.0;
	double currentOutputError = 0.0;
	for (int i = 0; i < numOutputs; i++)
	{
		outputErrors[i] = (outputs[i] - outputPredictions[i]);
		currentOutputError = abs(outputErrors[i]);
		sumCurrentExampleError += currentOutputError;
	}
	//update mean squared error
	sumMeanEpochOutputError += (sumCurrentExampleError  / numOutputs) * (sumCurrentExampleError  / numOutputs);

};


/************************************************************
Back Propagation Algorithm
NOTE: This method, given errors calculated in forward propagate,
updates the weights and delta weights.
************************************************************/
void NeuralNetwork::backPropagate()
{
	/**********************************
	Update local gradients and delta weights
	**********************************/
	//FUTURE WORK: In a more generalized version, this would be a for loop based on an aribtrary number of layers
	//      Where the number of loops is N - 1, where N = 1 input layer + 1 output layer + n hidden layers
	//FUTURE WORK: would allow for use of any activation function and its corresponding derivative

	//calculate local gradient for outputs using derivative of sigmoid function
	for (unsigned int i = 0 ; i < outputGradients.size() ; i++)
	{
		outputGradients[i] = lambda * outputPredictions[i] * (1 - outputPredictions[i]) * outputErrors[i];
	}

	//calculate delta weights for hidden to output layer
	for (unsigned int i = 0; i < hiddenToOutputCurrentDeltaWeights.size() ; i++)
	{
		for (unsigned int j = 0; j < hiddenToOutputCurrentDeltaWeights[i].size(); j++)
		{
			hiddenToOutputCurrentDeltaWeights[i][j] =
				eta * outputGradients[i] * hiddenPredictions[j] +
				alpha * hiddenToOutputPreviousDeltaWeights[i][j];
		}
	}

	//calculate hidden layer local gradients using derivative of sigmoid function
	// i = 1 so that we don't calculate derivative for bias (this is unecessary
	//TODO: Future work would allow for use of any activation function and its corresponding derivative
	for (unsigned int i = 1 ; i < hiddenGradients.size() ; i++)
	{
		double sumDeltaTimesWeights = 0.0;
		for (unsigned int j = 0; j < outputGradients.size() ; j++)
		{
			sumDeltaTimesWeights += outputGradients[j] * hiddenToOutputWeights[j][i];
		}
		hiddenGradients[i] = lambda * hiddenPredictions[i] * (1 - hiddenPredictions[i]) * sumDeltaTimesWeights;
	}

	//calculate delta weights for input to hidden layer
	for (unsigned int i = 0; i < inputToHiddenCurrentDeltaWeights.size() ; i++)
	{
		for (unsigned int j = 0; j < inputToHiddenCurrentDeltaWeights[i].size(); j++)
		{
			inputToHiddenCurrentDeltaWeights[i][j] =
				eta * hiddenGradients[i + 1] * inputs[j] +
				alpha * inputToHiddenPreviousDeltaWeights[i][j];
		}
	}

	/****************************
	Update neural network weights
	*****************************/
	//FUTURE WORK: consider whether it is best to have weight updating at the end of backpropagation or put it in a seperate method???
	//update input to hidden weights
	for (unsigned int i = 0; i < inputToHiddenWeights.size() ; i++)
	{
		for (unsigned int j = 0; j < inputToHiddenWeights[i].size(); j++)
		{
			inputToHiddenPreviousDeltaWeights[i][j] = inputToHiddenCurrentDeltaWeights[i][j];
			inputToHiddenWeights[i][j] += inputToHiddenCurrentDeltaWeights[i][j];
		}
	}

	//update hidden to output weights
	for (unsigned int i = 0; i < hiddenToOutputWeights.size() ; i++)
	{
		for (unsigned int j = 0; j < hiddenToOutputWeights[i].size(); j++)
		{
			hiddenToOutputPreviousDeltaWeights[i][j] = hiddenToOutputCurrentDeltaWeights[i][j];
			hiddenToOutputWeights[i][j] += hiddenToOutputCurrentDeltaWeights[i][j];
		}
	}
}

/************************************************************
Calculate the RMSE for the current online epoch (non Batch)
************************************************************/
double NeuralNetwork::calcRMSE(int numExamples)
{
	double rmse = 0.0;
	rmse = sqrt(sumMeanEpochOutputError / numExamples);
	sumMeanEpochOutputError = 0.0;
	return rmse;
}

/************************************************************
Denormalize the output
************************************************************/
void NeuralNetwork::denormOutputPredictions(std::vector<double> &robot){
	std::vector <double> denormedOutputs;
	denormedOutputs.reserve(numOutputs);
	for (unsigned int i = 0; i < outputPredictions.size(); i++) {
		denormedOutputs.push_back( (outputPredictions[i] * (outputMaxes[i] - outputMins[i]) ) + outputMins[i]);
	}

	for (unsigned int i = 0; i < outputPredictions.size(); i++) {
		robot[i] = denormedOutputs[i];
	}
};


/************************************************************
Remaining code in this class is necessary for batch update process for robot
************************************************************/

/************************************************************
Reset all current and previous delta weights
NOTE: The overall weights
************************************************************/
void NeuralNetwork::resetPreviousDeltaWeights(void){
	//reset input to hidden delta weights
	for (unsigned int i = 0; i < inputToHiddenWeights.size() ; i++)
	{
		for (unsigned int j = 0; j < inputToHiddenWeights[i].size(); j++)
		{
			inputToHiddenPreviousDeltaWeights[i][j] = 0.0;
			inputToHiddenCurrentDeltaWeights[i][j] = 0.0;
		}
	}

	//reset hidden to output delta weights
	for (unsigned int i = 0; i < hiddenToOutputWeights.size() ; i++)
	{
		for (unsigned int j = 0; j < hiddenToOutputWeights[i].size(); j++)
		{
			hiddenToOutputPreviousDeltaWeights[i][j] += 0.0;
			hiddenToOutputCurrentDeltaWeights[i][j] += 0.0;
		}
	}
};

/************************************************************
Set outputs to current values found to be closest to example in training data set
Also, sum the batch  squared error
************************************************************/
void NeuralNetwork::batchSetOutputs(std::vector <double> onlineTrainOutputs , std::vector <double> onlineTrainOutputsPreds){
	double sumCurrentExampleError = 0.0;
	double currentOutputError = 0.0;
	for (unsigned int i = 0; i < onlineTrainOutputs.size(); i ++){
		outputErrors[i] = (onlineTrainOutputs[i] - onlineTrainOutputsPreds[i]);
		outputPredictions[i] = onlineTrainOutputsPreds[i];
		currentOutputError = abs(outputErrors[i]);
		/*Used for RMSE of each output*/
		batchSumErrors[i] += currentOutputError * currentOutputError;   ///get square for current Output
		sumCurrentExampleError += currentOutputError;
	}
	/*Used for RMSE of the entire batch*/
	sumBatchError += (sumCurrentExampleError  / numOutputs) * (sumCurrentExampleError  / numOutputs);
}

/************************************************************
Runs at the end of each Batch learningepoch,
1) calculates the RMSE for the entire batch
2) sets the output Error for each wheel to the RMSE for that wheel
3) resets batchSumErrors
************************************************************/
double NeuralNetwork::batchLearn(int numOfExamples){
	double rmseBatch = 0.0;
	rmseBatch = sqrt(sumBatchError / numOfExamples);
	sumBatchError = 0.0;

	for (int i = 0; i < numOutputs; i++)
	{
		outputErrors[i] = sqrt(batchSumErrors[i] / numOfExamples);

		//reset batchSumErrors
		batchSumErrors[i] = 0.0;
	}
	return rmseBatch;
}