package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/infiniteCrank/mathbot/NeuralNetwork" // import your custom package
)

func main() {
	// For reproducible random initialization
	rand.Seed(time.Now().UnixNano())

	// Training data: sequences of five numbers and their following number.
	// For example, [1,2,3,4,5] should predict 6.
	trainingInputs := [][]float64{
		{1, 2, 3, 4, 5},
		{2, 3, 4, 5, 6},
		{3, 4, 5, 6, 7},
		{4, 5, 6, 7, 8},
		{5, 6, 7, 8, 9},
	}
	trainingTargets := [][]float64{
		{6},
		{7},
		{8},
		{9},
		{10},
	}

	// Create a new neural network.
	// We define the network with layer sizes: 5 -> 10 -> 1.
	// For activations, we choose:
	//   - Hidden layer: ReLUActivation
	//   - Output layer: LeakyReLUActivation (acts nearly linear for positive numbers)
	nn := NeuralNetwork.NewNeuralNetwork(
		[]int{5, 10, 1},
		[]int{NeuralNetwork.ReLUActivation, NeuralNetwork.LeakyReLUActivation},
		0.01,  // learning rate
		0.001, // L2 regularization factor
	)

	// Train the network.
	// Here we train for 1000 iterations, decaying the learning rate by 0.99 every 100 epochs.
	iterations := 1000
	learningRateDecayFactor := 0.99
	decayEpochs := 100

	fmt.Println("Training started...")
	nn.Train(trainingInputs, trainingTargets, iterations, learningRateDecayFactor, decayEpochs)
	fmt.Println("Training completed.")

	// Now test the network.
	// For the sequence [6,7,8,9,10] the expected prediction is 11.
	testInput := []float64{6, 7, 8, 9, 10}
	predicted := nn.PredictRegression(testInput)

	fmt.Printf("For the input sequence %v, the network predicts: %v\n", testInput, predicted)
}
