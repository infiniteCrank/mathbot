package NeuralNetwork

import "math"

// CalculateMSE computes the mean squared error for the full vectors.
func CalculateMSE(outputs []float64, targets []float64) float64 {
	mse := 0.0
	for i := 0; i < len(outputs) && i < len(targets); i++ {
		mse += 0.5 * math.Pow(targets[i]-outputs[i], 2)
	}
	return mse
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
