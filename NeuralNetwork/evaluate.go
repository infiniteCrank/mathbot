package NeuralNetwork

import "math"

// CalculateMSE computes the Mean Squared Error between predictions and targets.
func CalculateMSE(predictions, targets []float64) float64 {
	mse := 0.0
	for i := range predictions {
		mse += math.Pow(targets[i]-predictions[i], 2)
	}
	return mse / float64(len(predictions))
}

// CalculateRMSE computes the Root Mean Squared Error.
func CalculateRMSE(predictions, targets []float64) float64 {
	return math.Sqrt(CalculateMSE(predictions, targets))
}

// CalculateMAE computes the Mean Absolute Error.
func CalculateMAE(predictions, targets []float64) float64 {
	mae := 0.0
	for i := range predictions {
		mae += math.Abs(targets[i] - predictions[i])
	}
	return mae / float64(len(predictions))
}

// Evaluate model accuracy against a target threshold
func EvaluateAccuracy(nn *NeuralNetwork, testData [][]float64, targetData [][]float64, threshold float64, maxValue float64) float64 {
	correct := 0
	for i := range testData {
		prediction := nn.PredictRegression(testData[i])
		predictedValue := prediction[0] * maxValue // Rescale back
		actualValue := targetData[i][0] * maxValue // Rescale back

		if math.Abs(predictedValue-actualValue) < threshold {
			correct++
		}
	}
	return float64(correct) / float64(len(testData)) * 100.0 // Return accuracy percentage
}
