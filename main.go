package main

import (
	"fmt"

	"github.com/infiniteCrank/mathbot/NeuralNetwork" // import your custom package
)

// Define an identity activation constant. Make sure your package's Train method handles this case.
const IdentityActivation = 100

func main() {

	// Original training data: sequences of five numbers predicting the next number.
	// For example, [1,2,3,4,5] -> 6, etc.
	trainingInputs := [][]float64{
		{1, 2, 3, 4, 5},
		{2, 3, 4, 5, 6},
		{3, 4, 5, 6, 7},
		{4, 5, 6, 7, 8},
		{5, 6, 7, 8, 9},
		{6, 7, 8, 9, 10},
		{7, 8, 9, 10, 11},
		{8, 9, 10, 11, 12},
		{9, 10, 11, 12, 13},
		{10, 11, 12, 13, 14},
		{11, 12, 13, 14, 15},
		{12, 13, 14, 15, 16},
		{13, 14, 15, 16, 17},
		{14, 15, 16, 17, 18},
		{15, 16, 17, 18, 19},
	}
	trainingTargets := [][]float64{
		{6},
		{7},
		{8},
		{9},
		{10},
		{11},
		{12},
		{13},
		{14},
		{15},
		{16},
		{17},
		{18},
		{19},
		{20},
	}

	// Create a new neural network.
	// Architecture: 5 input neurons, one hidden layer with 10 neurons, and 1 output neuron.
	// We use ReLU for the hidden layer and Identity for the output layer.
	nn := NeuralNetwork.NewNeuralNetwork(
		[]int{5, 10, 1},
		[]int{NeuralNetwork.ReLUActivation, IdentityActivation},
		0.001, // Lower learning rate.
		0.001, // L2 regularization factor.
	)

	// Training parameters.
	iterations := 100000
	learningRateDecayFactor := 0.99
	decayEpochs := 1000
	miniBatchSize := 5 // With 5 training samples, process them all together.

	fmt.Println("Training started...")
	nn.Train(trainingInputs, trainingTargets, iterations, learningRateDecayFactor, decayEpochs, miniBatchSize)
	fmt.Println("Training completed.")

	// Test the network with a new sequence: [6,7,8,9,10].
	testInput := []float64{6, 7, 8, 9, 10}
	// Normalize test input (same as training data).
	for i := range testInput {
		testInput[i] /= 10.0
	}

	// Use your regression prediction method (assumed to be provided in your package)
	predicted := nn.PredictRegression(testInput)
	// Scale the prediction back.
	predicted[0] *= 10.0

	fmt.Printf("For the input sequence %v, the network predicts: %v\n", []float64{6, 7, 8, 9, 10}, predicted)
}
