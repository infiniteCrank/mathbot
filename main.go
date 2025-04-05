package main

import (
	"fmt"
	"math/rand"

	"github.com/infiniteCrank/mathbot/NeuralNetwork"
)

func main() {
	// Increase training range
	startNum := 1
	endNum := 6000 // More training samples

	var trainingInputs [][]float64
	var trainingTargets [][]float64

	// Introduce variability in sequence generation
	for i := startNum; i <= endNum-5; i++ {
		sequence := []float64{
			float64(i) + rand.Float64()*0.1,
			float64(i+1) + rand.Float64()*0.1,
			float64(i+2) + rand.Float64()*0.1,
			float64(i+3) + rand.Float64()*0.1,
			float64(i+4) + rand.Float64()*0.1,
		}
		trainingInputs = append(trainingInputs, sequence)
		trainingTargets = append(trainingTargets, []float64{float64(i + 5)})
	}

	// Normalize inputs and targets.
	maxValue := float64(endNum + 5)
	for i := range trainingInputs {
		for j := range trainingInputs[i] {
			trainingInputs[i][j] /= maxValue
		}
	}
	for i := range trainingTargets {
		trainingTargets[i][0] /= maxValue
	}

	fmt.Printf("Generated %d training examples.\n", len(trainingInputs))

	// Set initial hyperparameters.
	learningRate := 0.001 // Increase the learning rate for faster convergence
	l2Regularization := 0.001

	// Create a neural network with an adjusted architecture.
	// Note: Use NeuralNetwork.IdentityActivation (from your package) for regression.
	nn := NeuralNetwork.NewNeuralNetwork(
		[]int{5, 25, 50, 10, 1}, // Adjusted layers
		[]int{NeuralNetwork.LeakyReLUActivation, NeuralNetwork.LeakyReLUActivation, NeuralNetwork.LeakyReLUActivation, NeuralNetwork.IdentityActivation},
		learningRate,
		l2Regularization,
	)

	// Load the saved weights if available.
	fmt.Println("Loading weights...")
	nn.LoadWeights()
	fmt.Println("Weights loaded.")

	fmt.Println("Training started...")
	// Train with more iterations. Note: The patience mechanism in Train will handle early stopping.
	nn.Train(trainingInputs, trainingTargets, 20000, 0.9999, 5000, 32)
	fmt.Println("Training completed.")

	// Evaluate the network on test data.
	var predictions []float64
	var actuals []float64

	for i := 1001; i <= 2020; i++ { // Test on a small range outside training data
		testInput := []float64{float64(i), float64(i + 1), float64(i + 2), float64(i + 3), float64(i + 4)}
		for j := range testInput {
			testInput[j] /= maxValue // Normalize for input
		}

		predicted := nn.PredictRegression(testInput)
		predicted[0] *= maxValue // Rescale back

		actual := float64(i + 5)
		predictions = append(predictions, predicted[0])
		actuals = append(actuals, actual)
	}

	// Compute regression metrics.
	mse := NeuralNetwork.CalculateMSE(predictions, actuals)
	rmse := NeuralNetwork.CalculateRMSE(predictions, actuals)
	mae := NeuralNetwork.CalculateMAE(predictions, actuals)

	fmt.Printf("Evaluation Metrics:\nMSE: %.6f, RMSE: %.6f, MAE: %.6f\n", mse, rmse, mae)

	// Print sample predictions.
	fmt.Println("Sample Predictions vs Actual Values:")
	for i := 0; i < min(5, len(predictions)); i++ {
		fmt.Printf("Predicted: %.4f, Actual: %.4f\n", predictions[i], actuals[i])
	}

	// Decide whether to save weights based on a regression metric (e.g., RMSE).
	// Here we save the weights only if the RMSE is below a chosen threshold.
	rmseThreshold := 50.0
	if rmse < rmseThreshold {
		fmt.Println("RMSE is acceptable, saving weights...")
		nn.DeleteWeights() // Remove old weights
		nn.SaveWeights()   // Save new weights
		fmt.Println("Weights saved.")
	} else {
		fmt.Println("RMSE is high; not saving weights.")
	}
}

// Helper function to get the minimum of two values.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
