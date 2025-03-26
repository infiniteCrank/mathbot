package main

import (
	"fmt"

	"github.com/infiniteCrank/mathbot/NeuralNetwork" // import your custom package
)

// Define an identity activation constant.
// Ensure your package handles IdentityActivation in the backward pass.
const IdentityActivation = 100

func main() {

	// Generate training data dynamically.
	// Let's create sequences from 1 to 50, so that each input has five numbers and the target is the next number.
	var trainingInputs [][]float64
	var trainingTargets [][]float64
	startNum := 1
	endNum := 50

	// We need to ensure that i+5 is within our desired range.
	for i := startNum; i <= endNum-5; i++ {
		// Create an input sequence of five consecutive numbers.
		sequence := []float64{
			float64(i),
			float64(i + 1),
			float64(i + 2),
			float64(i + 3),
			float64(i + 4),
		}
		trainingInputs = append(trainingInputs, sequence)
		// The target is the next number.
		trainingTargets = append(trainingTargets, []float64{float64(i + 5)})
	}

	// Normalize inputs and targets.
	// For this example, we divide by 10 so that values are roughly between 0 and 10 become 0 and 1.
	for i := range trainingInputs {
		for j := range trainingInputs[i] {
			trainingInputs[i][j] /= 10.0
		}
	}
	for i := range trainingTargets {
		for j := range trainingTargets[i] {
			trainingTargets[i][j] /= 10.0
		}
	}

	// Print how many training examples we have.
	fmt.Printf("Generated %d training examples.\n", len(trainingInputs))

	// Create a new neural network.
	// Architecture: 5 input neurons, one hidden layer with 10 neurons, and 1 output neuron.
	// Use ReLU for the hidden layer and Identity for the output layer.
	nn := NeuralNetwork.NewNeuralNetwork(
		[]int{5, 10, 1},
		[]int{NeuralNetwork.ReLUActivation, IdentityActivation},
		0.0001, // Lower learning rate.
		0.001,  // L2 regularization factor.
	)

	// Training parameters.
	iterations := 5000000
	learningRateDecayFactor := 0.99
	decayEpochs := 1000
	miniBatchSize := len(trainingInputs) // Use full batch or set to a smaller number if desired.

	fmt.Println("Training started...")
	nn.Train(trainingInputs, trainingTargets, iterations, learningRateDecayFactor, decayEpochs, miniBatchSize)
	fmt.Println("Training completed.")

	// Test the network with a new sequence: [6, 7, 8, 9, 10].
	testInput := []float64{6, 7, 8, 9, 10}
	// Normalize test input.
	for i := range testInput {
		testInput[i] /= 10.0
	}

	// Use your regression prediction method from the package.
	predicted := nn.PredictRegression(testInput)
	// Scale the prediction back.
	predicted[0] *= 10.0

	fmt.Printf("For the input sequence %v, the network predicts: %v\n", []float64{6, 7, 8, 9, 10}, predicted)
}
