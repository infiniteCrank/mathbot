package NeuralNetwork

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/infiniteCrank/mathbot/tfidf"
)

// Activation Function Types
const (
	SigmoidActivation = iota
	ReLUActivation
	TanhActivation
	LeakyReLUActivation
	SoftmaxActivation // Add Softmax activation
)

// Activation functions

// Sigmoid activation function
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Derivative of the Sigmoid function
func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

// ReLU activation function
func relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

// Derivative of the ReLU function
func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// Tanh activation function
func tanh(x float64) float64 {
	return math.Tanh(x)
}

// Derivative of the Tanh function
func tanhDerivative(x float64) float64 {
	return 1 - x*x
}

// Leaky ReLU activation function
func leakyReLU(x float64) float64 {
	if x < 0 {
		return 0.01 * x // Leaky component
	}
	return x
}

// Derivative of the Leaky ReLU function
func leakyReLUDerivative(x float64) float64 {
	if x < 0 {
		return 0.01 // Leaky component
	}
	return 1
}

// Softmax activation function
func softmax(logits []float64) []float64 {
	expScores := make([]float64, len(logits))
	sumExpScores := 0.0

	for i, score := range logits {
		expScores[i] = math.Exp(score)
		sumExpScores += expScores[i]
	}

	for i := range expScores {
		expScores[i] /= sumExpScores // Normalize to get probabilities
	}

	return expScores
}

// Layer structure
type Layer struct {
	inputs     int         // Number of inputs to the layer
	outputs    int         // Number of outputs from the layer
	weights    [][]float64 // Weights connecting inputs to outputs
	biases     []float64   // Biases for the layer's outputs
	activation int         // Activation function used in this layer
	gamma      []float64   // Scale parameters for batch normalization
	beta       []float64   // Shift parameters for batch normalization
}

// NewLayer creates and initializes a new layer
func NewLayer(inputs, outputs int, activation int) *Layer {
	// Initialize weights with random values
	weights := make([][]float64, inputs)
	for i := range weights {
		weights[i] = make([]float64, outputs)
		for j := range weights[i] {
			weights[i][j] = rand.NormFloat64() // Random initialization
		}
	}

	// Initialize biases to random values
	biases := make([]float64, outputs)
	for i := range biases {
		biases[i] = rand.NormFloat64() // Random initialization
	}

	// Initialize gamma and beta for batch normalization
	gamma := make([]float64, outputs)
	beta := make([]float64, outputs)
	for i := range gamma {
		gamma[i] = 1.0 // Scale initialized to 1
		beta[i] = 0.0  // Shift initialized to 0
	}

	return &Layer{inputs, outputs, weights, biases, activation, gamma, beta}
}

// Forward pass with Batch Normalization
func (l *Layer) Forward(input []float64, training bool) ([]float64, []float64, []float64) {
	outputs := make([]float64, l.outputs) // Initialize layer outputs
	var batchMeans []float64              // Holds mean values for batch normalization
	var batchVariances []float64          // Holds variance values for batch normalization

	// Calculate the weighted input and biases
	for j := 0; j < l.outputs; j++ {
		sum := 0.0
		for i := 0; i < l.inputs; i++ {
			sum += input[i] * l.weights[i][j] // Weighted sum
		}
		sum += l.biases[j] // Apply bias
		outputs[j] = sum   // Store result in outputs
	}

	if training {
		// If we are in training mode, compute batch normalization statistics
		batchMeans = make([]float64, l.outputs)
		batchVariances = make([]float64, l.outputs)

		for j := range outputs {
			batchMeans[j] = outputs[j] / float64(l.outputs)                                              // Mean for the batch
			batchVariances[j] = (outputs[j]*outputs[j])/float64(l.outputs) - batchMeans[j]*batchMeans[j] // Variance for the batch
		}

		// Normalize outputs using batch normalization
		for j := range outputs {
			outputs[j] = l.gamma[j]*(outputs[j]-batchMeans[j])/math.Sqrt(batchVariances[j]+1e-8) + l.beta[j]
		}

	} else {
		// For inference, ideally use a running average of means/variances, using last computed mean/variance
		for j := range outputs {
			outputs[j] = l.gamma[j]*(outputs[j]) + l.beta[j] // Adjust without normalization for inference
		}
	}

	// Return outputs, means, and variances
	return outputs, batchMeans, batchVariances
}

// Calculate the error for outputs
func calculateError(outputs []float64, targets []float64) float64 {
	error := 0.0
	for i := range outputs {
		error += 0.5 * math.Pow(targets[i]-outputs[i], 2) // Mean Squared Error
	}
	return error
}

// NeuralNetwork structure
type NeuralNetwork struct {
	layers           []*Layer
	learningRate     float64
	l2Regularization float64
}

// NewNeuralNetwork initializes a neural network
func NewNeuralNetwork(layerSizes []int, activations []int, learningRate float64, l2Regularization float64) *NeuralNetwork {
	nn := &NeuralNetwork{learningRate: learningRate, l2Regularization: l2Regularization}
	for i := 0; i < len(layerSizes)-1; i++ {
		nn.layers = append(nn.layers, NewLayer(layerSizes[i], layerSizes[i+1], activations[i]))
	}
	return nn
}

// Train the network with Batch Normalization and L2 regularization
func (nn *NeuralNetwork) Train(inputs [][]float64, targets [][]float64, iterations int,
	learningRateDecayFactor float64, decayEpochs int, miniBatchSize int) {

	totalSamples := len(inputs)
	for iter := 0; iter < iterations; iter++ {
		// Decay learning rate periodically.
		if iter > 0 && iter%decayEpochs == 0 {
			nn.learningRate *= learningRateDecayFactor
		}

		// Process mini-batches.
		for start := 0; start < totalSamples; start += miniBatchSize {
			end := start + miniBatchSize
			if end > totalSamples {
				end = totalSamples
			}
			// Create mini-batch slices.
			batchInputs := inputs[start:end]   // [][]float64: batchSize x inputDimension
			batchTargets := targets[start:end] // [][]float64: batchSize x targetDimension
			batchSize := len(batchInputs)

			// Allocate storage for outputs per layer.
			// outputs[0] holds the mini-batch input.
			outputs := make([][][]float64, len(nn.layers)+1)
			outputs[0] = batchInputs

			// To store batch normalization statistics (if needed).
			batchMeans := make([][]float64, len(nn.layers))
			batchVariances := make([][]float64, len(nn.layers))

			// Forward pass: propagate mini-batch through each layer.
			for k := 1; k <= len(nn.layers); k++ {
				out, means, variances := nn.layers[k-1].ForwardBatch(outputs[k-1], true)
				outputs[k] = out
				batchMeans[k-1] = means
				batchVariances[k-1] = variances
			}

			// Final outputs from the network for this mini-batch.
			finalOutputs := outputs[len(nn.layers)] // shape: batchSize x outputDimension

			// Compute errors for the output layer for each sample.
			errors := make([][][]float64, len(nn.layers))
			errors[len(nn.layers)-1] = make([][]float64, batchSize)
			for b := 0; b < batchSize; b++ {
				errVec := make([]float64, len(finalOutputs[b]))
				for j := 0; j < len(finalOutputs[b]); j++ {
					errVec[j] = batchTargets[b][j] - finalOutputs[b][j]
				}
				errors[len(nn.layers)-1][b] = errVec
			}

			// Backward pass: propagate errors from the last layer back to the first.
			for l := len(nn.layers) - 1; l >= 0; l-- {
				// For each neuron j in layer l, compute the average gradient over the mini-batch.
				for j := 0; j < nn.layers[l].outputs; j++ {
					var gradientSum float64 = 0.0
					for b := 0; b < batchSize; b++ {
						var grad float64
						// Use the activation derivative on the output from this layer.
						switch nn.layers[l].activation {
						case SigmoidActivation:
							grad = sigmoidDerivative(outputs[l+1][b][j])
						case ReLUActivation:
							grad = reluDerivative(outputs[l+1][b][j])
						case TanhActivation:
							grad = tanhDerivative(outputs[l+1][b][j])
						case LeakyReLUActivation:
							grad = leakyReLUDerivative(outputs[l+1][b][j])
						case SoftmaxActivation:
							grad = outputs[l+1][b][j] * (1 - outputs[l+1][b][j])
						}
						gradientSum += errors[l][b][j] * grad
					}
					avgGradient := gradientSum / float64(batchSize)
					// Update weights for neuron j in layer l.
					for i := 0; i < nn.layers[l].inputs; i++ {
						var deltaSum float64 = 0.0
						for b := 0; b < batchSize; b++ {
							deltaSum += avgGradient * outputs[l][b][i]
						}
						avgDelta := (deltaSum / float64(batchSize)) * nn.learningRate
						// Weight update with L2 regularization.
						nn.layers[l].weights[i][j] += avgDelta
						nn.layers[l].weights[i][j] -= nn.l2Regularization * nn.learningRate * nn.layers[l].weights[i][j]
					}
					// Update bias and batch normalization parameters for neuron j.
					nn.layers[l].biases[j] += avgGradient * nn.learningRate
					nn.layers[l].gamma[j] += avgGradient * nn.learningRate
					nn.layers[l].beta[j] += avgGradient * nn.learningRate
				}

				// Propagate error to the previous layer if this is not the first layer.
				if l > 0 {
					errors[l-1] = make([][]float64, batchSize)
					for b := 0; b < batchSize; b++ {
						prevErr := make([]float64, nn.layers[l-1].outputs)
						for i := 0; i < nn.layers[l-1].outputs; i++ {
							sumError := 0.0
							for j := 0; j < nn.layers[l].outputs; j++ {
								sumError += errors[l][b][j] * nn.layers[l].weights[i][j]
							}
							prevErr[i] = sumError
						}
						errors[l-1][b] = prevErr
					}
				}
			} // End of backward pass.
		} // End of mini-batch loop.
	} // End of iterations.
}

// K-Fold Cross Validation Function
func performKFoldCrossValidation(nn *NeuralNetwork, inputs [][]float64, targets [][]float64,
	k int, iterations int, learningRateDecayFactor float64, decayEpochs int) {

	foldSize := len(inputs) / k
	var totalValidationLoss float64

	for i := 0; i < k; i++ {
		// Split the dataset into validation and training sets
		validationInputs := inputs[i*foldSize : (i+1)*foldSize]
		validationTargets := targets[i*foldSize : (i+1)*foldSize]

		// Combine other folds for training
		trainingInputs := append(inputs[:i*foldSize], inputs[(i+1)*foldSize:]...)
		trainingTargets := append(targets[:i*foldSize], targets[(i+1)*foldSize:]...)

		// Train the model on training set
		nn.Train(trainingInputs, trainingTargets, iterations, learningRateDecayFactor, decayEpochs, 5)

		// Evaluate the model
		validationLoss := 0.0
		for j := range validationInputs {
			valOutput := nn.Predict(validationInputs[j])
			validationLoss += calculateError(valOutput, validationTargets[j]) // Compute validation loss
		}
		validationLoss /= float64(len(validationInputs))
		totalValidationLoss += validationLoss
		fmt.Printf("Validation Loss for fold %d: %.6f\n", i+1, validationLoss)
	}

	averageValidationLoss := totalValidationLoss / float64(k)
	fmt.Printf("Average Validation Loss across all folds: %.6f\n", averageValidationLoss)
}

// Predict using the neural network
func (nn *NeuralNetwork) Predict(input []float64) []float64 {
	for i, layer := range nn.layers { // The Forward method returns the outputs, batch means, and batch variances.
		// In the Predict function, we should only be interested in the outputs:
		input, _, _ = layer.Forward(input, false)
		if i == len(nn.layers)-1 {
			input = softmax(input) // Apply softmax for probabilities on the output layer
		}
	}
	return input
}

// PrepareInputOutput prepares the input-output pairs from sentences to predict next words.
func PrepareInputOutput(corpus []string, tfidfModel *tfidf.TFIDF) ([][]float64, [][]float64) {
	var inputs [][]float64
	var targets [][]float64

	for _, sentence := range corpus {
		words := strings.Fields(sentence)
		for i := 0; i < len(words)-1; i++ {
			// Get the current word as input and the next word as target
			currentWord := words[i]
			nextWord := words[i+1]

			// Get the TF-IDF vector for the current word
			tfIdfScores := tfidfModel.CalculateScores() // This populates tfidfModel.Scores
			input := make([]float64, len(tfidfModel.ProcessedWords))
			if score, exists := tfIdfScores[currentWord]; exists {
				input[tfidfModel.ProcessedWordsIndex(currentWord)] = score // Get the TF-IDF score for current word
			}
			inputs = append(inputs, input)

			// Prepare the target as one-hot encoding of the next word
			target := make([]float64, len(tfidfModel.ProcessedWords))
			if score, exists := tfIdfScores[nextWord]; exists {
				target[tfidfModel.ProcessedWordsIndex(nextWord)] = score // Get the TF-IDF score for next word
			}
			targets = append(targets, target)
		}
	}

	return inputs, targets
}

// PredictRegression performs a forward pass without applying softmax to the final layer.
func (nn *NeuralNetwork) PredictRegression(input []float64) []float64 {
	output := input
	for _, layer := range nn.layers { // Accessing the unexported field is OK within the package.
		// Only use the output from Forward.
		output, _, _ = layer.Forward(output, false)
	}
	return output
}

func (l *Layer) ForwardBatch(inputs [][]float64, training bool) ([][]float64, []float64, []float64) {
	batchSize := len(inputs)
	// Allocate outputs: each sample will produce a slice of length l.outputs.
	outputs := make([][]float64, batchSize)
	for b, input := range inputs {
		out := make([]float64, l.outputs)
		for j := 0; j < l.outputs; j++ {
			sum := 0.0
			for i := 0; i < l.inputs; i++ {
				sum += input[i] * l.weights[i][j]
			}
			sum += l.biases[j]
			out[j] = sum
		}
		outputs[b] = out
	}

	var batchMeans, batchVariances []float64
	if training {
		batchMeans = make([]float64, l.outputs)
		batchVariances = make([]float64, l.outputs)
		// Compute mean for each neuron over the batch.
		for j := 0; j < l.outputs; j++ {
			sum := 0.0
			for b := 0; b < batchSize; b++ {
				sum += outputs[b][j]
			}
			mean := sum / float64(batchSize)
			batchMeans[j] = mean
		}
		// Compute variance for each neuron over the batch.
		for j := 0; j < l.outputs; j++ {
			sumSq := 0.0
			for b := 0; b < batchSize; b++ {
				diff := outputs[b][j] - batchMeans[j]
				sumSq += diff * diff
			}
			variance := sumSq / float64(batchSize)
			batchVariances[j] = variance
		}
		// Normalize outputs using batch statistics.
		for b := 0; b < batchSize; b++ {
			for j := 0; j < l.outputs; j++ {
				outputs[b][j] = l.gamma[j]*(outputs[b][j]-batchMeans[j])/math.Sqrt(batchVariances[j]+1e-8) + l.beta[j]
			}
		}
	} else {
		// In inference mode, simply adjust with gamma and beta.
		for b := 0; b < batchSize; b++ {
			for j := 0; j < l.outputs; j++ {
				outputs[b][j] = l.gamma[j]*outputs[b][j] + l.beta[j]
			}
		}
	}

	return outputs, batchMeans, batchVariances
}
