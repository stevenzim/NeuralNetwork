#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

#include <Aria.h> //AriaDebugVC10.lib
#include <stdio.h>
#include <conio.h>
#include <time.h>

using namespace std;

int main( int argc, char** argv ) 
{
	/*******************************
	OFFLINE TRAINING NEURAL NETWORK
	********************************/

	cout << "OFFLINE TRAINING WILL BEGIN.  \nTHIS WILL LAST UNTIL STOPPING CRITERIA MET (1000 EPOCHS MAX)" << endl;
	system ("PAUSE");
	/*inputs and outputs, functionality easily handles different values*/
	int numInputs = 2;
	int numOutputs = 2;

	/*These are approximate max and min values for robot and dataset*/
	double inputMaxes[2] = {5500.0 ,5500.0};
	double inputMins[2] = {0,0};
	double outputMaxes[2] = {300.0,300.0};
	double outputMins[2] = {0,0};

	/*************************
	Read in training data
	**************************/
	std::ifstream iFile("test.csv");   //training data file

	string feature; // string to store current element of file

	/*Initiliaze 2D vectors to store train, validation and test examples.
	FUTURE WORK: Determine the necessity of the shuffle*/
	std::vector< std::vector<double> >trainData;
	std::vector< std::vector<double> >validData;
	std::vector< std::vector<double> >testData;

	/* read in file element 1&2 = inputs, element 3&4 = outputs*/
	int linecount = 0;
	while ( iFile.good() )
	{
		std::vector<double> row;
		row.resize(4);
		getline (iFile , feature , ',' );
		row[0] = stod(feature);
		getline (iFile , feature , ',' );
		row[1] = stod(feature);
		getline (iFile , feature , ',' );
		row[2] = stod(feature);
		getline (iFile , feature, '\n');
		row[3] = stod(feature);


		/*splitting of 1165 unique examples randomly shuffled in input file
		this is done to ensure an exact 70/15/15 split*/
		if ( linecount <= 815){      //first 70% to validation
			trainData.push_back(row);
		} else if (linecount <= 990) {   //next 15% to test
			validData.push_back(row);
		} else {
			testData.push_back(row);
		}
		linecount++;
	}
	iFile.close();

	/*shuffle the training validation and test sets to randomize data further*/
	std::random_shuffle(trainData.begin(),trainData.end());
	std::random_shuffle(validData.begin(),validData.end());
	std::random_shuffle(testData.begin(),testData.end());

	/***********************************************
	Initialize Robot Brain (i.e. Neural Network)
	************************************************/
	/*
	Tuning of NN Params - Per discussion in lecture, these paramaters were initially started 
	neurons = 2 * numInputs = 4
	eta = .7, alpha =.1 & lambda = .7
	All settings were determined empirically
	1) The training and validation graphs of RMSE vs Epoch do not behave as expected
	i.e. even after 10000 iterations the validation error does not increase
	perhaps the number of unique examples is to small, nonetheless the robot performs well
	2) ideal number of neurons was determined to be 7.  This was done by plotting RMSE for training and validation sets
	over 3000 epochs.  The lowest RMSE occured at 7 nodes.  Tested over a range of 4 to 15 nodes
	3) eta, the learning rate was best determined to be 1.5.  This was done incrimentally between
	.7 and 1.9 for 1000 epochs each.  RMSE was minimum at 1.5.
	4) alpha - momentum is necessary for controlling the speed of weight updates.  This value will fall between
	0 and 1.  I moved incrementally from .1 to .9 by steps of .1 for 1000 epochs.  .8 gave the lowest RMSE
	5) lambda - This value controls the steepness of the chosen regularization function (sigmoid).  .7 was the suggested
	starting value,  I chose to rn a test incrementally .1 stepwise between .5 and 1.0 for 1000 epochs.  .8 had best RMSE
	*/
	int neurons = 7;
	double eta = 1.5;
	double alpha = 0.9;
	double lambda = 0.8;

	NeuralNetwork robotBrain(2,neurons,2,eta,alpha,lambda);  //create neural net object
	robotBrain.initializeWeightMatrices();					//create random weights
	robotBrain.setNormalizationParams(inputMaxes,inputMins,outputMaxes,outputMins);  

	/*************************
	Initialize results file 
	**************************/
	//open results file   & create header
	std::string fname = "Offline-Results.csv";
	std::ofstream oFile(fname);
	oFile << "Epoch,RMSE_Training,RMSE_Validation" << endl;

	/*vars for stopping criteria*/
	double currentValidRMSE;
	double prevValidRMSE = 1000.0;
	double threshold = 0.000005;
	int numEpochsThresholdMet = 0;

	/*****************************************************************
	Teach the Robot with Neural Net
	For each epoch:
	1) Train with feedforward and backpropagation
	2) Get validation results
	3) If validation results are good enough, 
	then no more epochs -> perform test and get the final results,
	otherwise -> next epoch with feedforward and backpropagation
	*******************************************************************/

	/* Go no more than 1000 epochs for training*/
	for(int epoch = 1; epoch < 1000; epoch ++){

		//Run stochastic(online) training for current epoch
		for(unsigned int j = 0; j < trainData.size(); j++){
			robotBrain.feedForward(trainData[j]);
			robotBrain.backPropagate();
		}
		//output RMSE for training data to file
		oFile << epoch << "," << robotBrain.calcRMSE(trainData.size()) << "," ;

		/*Run feedforward on validation set to calculate RMSE*/
		for(unsigned int j = 0; j < validData.size(); j++){
			robotBrain.feedForward(validData[j]);
		}

		/*store and output RMSE validation data to file*/
		currentValidRMSE = robotBrain.calcRMSE(validData.size());
		oFile << currentValidRMSE << endl;

		/*print the number of epochs complete*/
		cout << "Epoch " << epoch << " complete." << endl;

		/*Here we assume that 5 epochs where the RMSE difference is less than threshold
		means that the change in RMSE is now insignificent.  With threshold set to .000005
		and num epochs  where threshold met is 5, then the total change is less than .000025.*/
		if ((prevValidRMSE - currentValidRMSE) <= threshold){
			numEpochsThresholdMet++;
		} else {
			numEpochsThresholdMet = 0;
		}

		/*If threshold met, then stop training*/
		/*if stopping criteria not met, then continue on*/
		if (numEpochsThresholdMet == 5){
			break;
		}

		/*update previous RMSE*/
		prevValidRMSE = currentValidRMSE;
	}

	/*Run feedforward on test set to calculate test set RMSE*/
	for(unsigned int j = 0; j < testData.size(); j++){
		robotBrain.feedForward(testData[j]);
	}
	/*output RMSE for test data*/
	oFile << "Test Set RMSE: " << robotBrain.calcRMSE(testData.size()) << endl;
	oFile.close();

	/***************************************** 
	INITIALIZE ROBOT  (CODE FROM CE889 LAB 1)
	*****************************************/

	cout << "INITIAL TRAINING COMPLETE!!!  ROBOT WILL NOW INITIALIZE" << endl;
	system ("PAUSE");

	/*robot and devices*/
	ArRobot robot;
	ArSonarDevice sonar;
	ArSick sick; 

	/*the laser
	connection*/
	ArDeviceConnection *con;

	/* Laser connection*/
	ArSerialConnection laserCon;

	/* add it to the robot*/
	robot.addRangeDevice(&sick);
	ArSerialConnection *serCon;
	serCon = new ArSerialConnection;
	serCon->setPort();

	/*serCon->setBaud(38400);*/
	con = serCon;

	/* set the connection on the robot*/
	robot.setDeviceConnection(con);

	/*initialize aria and aria's logging destination and level*/
	Aria::init();
	ArLog::init(ArLog::StdErr, ArLog::Normal);

	/*parser for the command line arguments*/
	ArArgumentParser parser(&argc, argv);
	parser.loadDefaultArguments();

	/*Connect to the robot, then load parameter files for this robot.*/
	ArRobotConnector robotConnector(&parser, &robot);
	if (!robotConnector.connectRobot()) {
		ArLog::log(ArLog::Terse, " It could not connect to the robot.");
		if (parser.checkHelpAndWarnUnparsed()) {
			// help not given
			Aria::logOptions();
			Aria::exit(1);
		}
	}
	if (!Aria::parseArgs()) {
		Aria::logOptions();
		Aria::shutdown();
		return 1;
	}
	ArLog::log(ArLog::Normal, "Program is Connected.");
	robot.setVel2(0,0); //safe start
	/*turn on the motors, turn off amigobot sounds*/
	robot.enableMotors();
	robot.comInt(ArCommands::SOUNDTOG, 0);

	/* start the robot running, true so that if we lose connection the run stops*/
	robot.runAsync(true);

	/* set up the port for the laser*/
	laserCon.setPort(ArUtil::COM3);
	sick.setDeviceConnection(&laserCon);

	/* now that we're connected to the robot, connect to the laser*/
	sick.runAsync();
	if (!sick.blockingConnect()) {
		printf("Could not connect to SICK laser... exiting\n");
		robot.disconnect();
		Aria::shutdown();
		return 1;
	}
	printf("Connected\n");
	ArUtil::sleep(500);
	/***********ROBOT INITIALIZED***************/


	/*********************************
	BEGIN ONLINE/BATCH LEARNING FOR ROBOT
	**********************************/

	cout << "ROBOT INITIALIZED, NOW RUN 3 EPOCHS FOR 'ONLINE' TRAINING" << endl;
	system ("PAUSE");

	/***********************************
	Initialize Online Training
	************************************/

	/*Flush out previous delta & current delta weights*/
	robotBrain.resetPreviousDeltaWeights();

	/*
	vecs for storing robot data.
	robotVals -> can hold inputs or predicted outputs
	robotOnlineExample -> holds the entire input and predicted output data
	*/
	std::vector<double> robotVals;
	std::vector<double> robotOnlineExample;
	robotVals.resize(2);
	robotOnlineExample.resize(4);

	std::vector <double> onlineTrainOutputs;
	std::vector <double> onlineTrainOutputsPreds;
	onlineTrainOutputs.resize(2);
	onlineTrainOutputsPreds.resize(2);

	std::ofstream onlineFile("OnlineTrainingErrors.csv");


	/*Batch learn for 3 Epochs*/
	for (int epoch = 1; epoch < 4; epoch++){
		/*start timer from 0 seconds, each epoch runs 15 secs*/
		time_t start = time(0);
		double secondsRun = difftime( time(0), start);

		/*count number of examples, necessary to calculate RMSE*/
		int numOfOnlineExamples = 0;

		/*Vars used to find the nearest training example to current robot laser reading*/
		double leftDistDiff = 0.0;
		double frontDistDiff = 0.0;
		double leftWheelDiff = 0.0;
		double rightWheelDiff = 0.0;
		double leftDistMin = 32001.0;
		double frontDistMin = 32001.0;
		double leftWheelDiffMin = 0.0;
		double rightWheelDiffMin = 0.0;

		while(secondsRun < 15)
		{
			/*update and output time*/
			secondsRun = difftime( time(0), start);
			cout << secondsRun << " seconds complete." << endl;

			/*Get current laser readings*/
			robotVals[0] = sick.currentReadingPolar(70,90);
			robotVals[1] = sick.currentReadingPolar(-15,15);

			/*If either val = 32000 then reading is an error
			If less than these values, then 
			1) get predicted wheel speeds with feed forward
			2) Find nearest example in training set
			3) Once closest example is found, we update outputs in the robotBrain
			and update the sum of errors*/
			if (robotVals[0] <32000 || robotVals[1] <32000){

				/*store robot readings in size 4 vector to be passed into feedforward*/
				robotOnlineExample[0] = robotVals[0];
				robotOnlineExample[1] = robotVals[1];

				/*pass size 2 vector of inputs into feedforward to get predicted outputs*/
				robotBrain.feedForward(robotVals);

				/*converts robot readings to wheel outputs by reference*/
				robotBrain.denormOutputPredictions(robotVals);

				/*store robot wheel speeds in size 4 vector to be passed into feedforward*/
				robotOnlineExample[2] = robotVals[0];
				robotOnlineExample[3] = robotVals[1];

				/*update robot speed*/
				robot.setVel2(robotVals[0],robotVals[1]);

				/*Find the closest example in training data, to calculate error
				NOTE: We assume that this nearest example is in fact a correct 
				representation. */
				for (unsigned int i = 0; i < trainData.size(); i++){
					/*Calculate diffirences for testing to determine if we have found
					an acceptably close enough example in training set*/
					leftDistDiff = abs(robotOnlineExample[0] - trainData[i][0]);
					frontDistDiff = abs(robotOnlineExample[1] - trainData[i][1]);
					leftWheelDiff = (robotOnlineExample[2] - trainData[i][2]);
					rightWheelDiff = (robotOnlineExample[3] - trainData[i][3]);

					/*Normalize the Predictions & Outputs*/
					onlineTrainOutputsPreds[0] = (robotOnlineExample[2] / outputMaxes[0]);
					onlineTrainOutputsPreds[1] = (robotOnlineExample[3] / outputMaxes[1]);
					onlineTrainOutputs[0] = (trainData[i][2] / outputMaxes[0]);
					onlineTrainOutputs[1] = (trainData[i][3] / outputMaxes[1]);

					/*If current example in training set has Left and Right laser readings
					closer than 1cm, then we assume this is representative
					otherwise we take the closest example possible*/
					if (leftDistDiff < 100.0 && frontDistDiff < 100.0){
						/*update robotBrain to have current outputs and predictions and to sum errors*/
						robotBrain.batchSetOutputs(onlineTrainOutputs, onlineTrainOutputsPreds);
						numOfOnlineExamples++;
						break;
					} else {
						/*if current diffs are less than mins, then update the mins*/
						if (leftDistDiff < leftDistMin && frontDistDiff < frontDistMin){
							leftDistMin = leftDistDiff;
							frontDistMin = frontDistDiff;
							leftWheelDiffMin = leftWheelDiff;
							rightWheelDiffMin = rightWheelDiff;
						}
						/*if we have reached the end training data set, use the minim*/
						if ((i + 1) == trainData.size()){
							/*update robotBrain to have current outputs and predictions and to sum errors*/
							robotBrain.batchSetOutputs(onlineTrainOutputs, onlineTrainOutputsPreds);
							numOfOnlineExamples++;
						}
					} 
				}
				/*reset distance mins*/
				leftDistMin = 32001.0;
				frontDistMin = 32001.0;
			}
		}

		/*Pause robot, let me know epoch is complete & output results to file*/
		robot.setVel2(0,0);
		cout << "Epoch " << epoch << " complete" << endl;
		onlineFile <<"Number of Examples this epoch: " << numOfOnlineExamples << endl;
		onlineFile << "RMSE this Epoch: " << robotBrain.batchLearn(numOfOnlineExamples) << endl;

		/*
		At this point we have error values for each wheel (RMSE for each wheel) where 
		RMSE = sqrt ( (sum of all (error of wheel ^2) in epoch ) / # of examples in epoch)
		Based on discussion in lecture these erros are basis for batch update.
		Also from lecture, we make an assumption that the last predictions for outputs &
		hidden layers are representative of all examples in batch update.
		We use the latest predicted values and overall RMSE values to run backpropagate 
		for the current batch.			
		*/
		robotBrain.backPropagate();
		system ("PAUSE");
	}
	onlineFile.close();


	/*************************************
	RUN FEEDFORWARD ON ROBOT 20 SECONDS
	*************************************/
	cout << "ONLINE/BATCH LEARNING COMPLETE, NOW ROBOT RUNS FOR 20 SECONDS" << endl;
	system ("PAUSE");

	time_t start = time(0);
	double secondsRun = difftime( time(0), start);
	while(secondsRun < 20)
	{
		/*start timer from 0 seconds, each epoch runs 20 secs*/
		secondsRun = difftime( time(0), start);
		cout << secondsRun << " seconds complete." << endl;

		/*Take laser readings*/
		robotVals[0] = sick.currentReadingPolar(70,90);
		robotVals[1] = sick.currentReadingPolar(-15,15);

		/*If readings are not errors, then
		1) run feedforward and get predictions
		2) denormalize predictions
		3) set Velocity with denormalized predictions
		*/
		if (robotVals[0] <32000 || robotVals[1] <32000){
			robotBrain.feedForward(robotVals);
			robotBrain.denormOutputPredictions(robotVals);
			robot.setVel2(robotVals[0],robotVals[1]);
		}
	}

	/************
	EXIT
	*************/
	cout << "20 SECONDS COMPLETE, PROGRAM WILL END." << endl;

	/*stop robot*/
	robot.setVel2(0,0);

	/*shutdown robot*/
	ArLog::log(ArLog::Normal, "Ending robot thread...");
	robot.stopRunning();
	robot.waitForRunExit(); // wait for robot task loop to end before exiting the program
	ArLog::log(ArLog::Normal, "Exiting."); //exit
	return 0;
}