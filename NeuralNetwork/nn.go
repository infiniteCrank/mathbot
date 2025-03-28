package NeuralNetwork

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Activation Function Types
const (
	SigmoidActivation = iota
	ReLUActivation
	TanhActivation
	LeakyReLUActivation
	SoftmaxActivation  // For classification tasks
	IdentityActivation // For regression tasks: output = weighted sum (i.e. linear)
)

// Activation functions

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

func tanhDerivative(x float64) float64 {
	return 1 - x*x
}

func leakyReLU(x float64) float64 {
	if x < 0 {
		return 0.01 * x
	}
	return x
}

func leakyReLUDerivative(x float64) float64 {
	if x < 0 {
		return 0.01
	}
	return 1
}

// Softmax activation for a vector input.
func softmax(logits []float64) []float64 {
	expScores := make([]float64, len(logits))
	sumExpScores := 0.0
	for i, score := range logits {
		expScores[i] = math.Exp(score)
		sumExpScores += expScores[i]
	}
	for i := range expScores {
		expScores[i] /= sumExpScores
	}
	return expScores
}

// Layer structure
type Layer struct {
	inputs     int
	outputs    int
	weights    [][]float64 // shape: inputs x outputs
	biases     []float64   // length: outputs
	activation int
	gamma      []float64 // for batch normalization (scale)
	beta       []float64 // for batch normalization (shift)
}

// NewLayer creates and initializes a new layer.
func NewLayer(inputs, outputs int, activation int) *Layer {
	weights := make([][]float64, inputs)
	for i := range weights {
		weights[i] = make([]float64, outputs)
		for j := range weights[i] {
			// Scale down the initialization to help training stability.
			weights[i][j] = rand.NormFloat64() * 0.01
		}
	}

	biases := make([]float64, outputs)
	for i := range biases {
		biases[i] = 0.0
	}

	gamma := make([]float64, outputs)
	beta := make([]float64, outputs)
	for i := range gamma {
		gamma[i] = 1.0
		beta[i] = 0.0
	}

	return &Layer{
		inputs:     inputs,
		outputs:    outputs,
		weights:    weights,
		biases:     biases,
		activation: activation,
		gamma:      gamma,
		beta:       beta,
	}
}

// Forward processes a single sample through the layer.
func (l *Layer) Forward(input []float64, training bool) ([]float64, []float64, []float64) {
	outputs := make([]float64, l.outputs)
	// Compute weighted sums.
	for j := 0; j < l.outputs; j++ {
		sum := 0.0
		for i := 0; i < l.inputs; i++ {
			sum += input[i] * l.weights[i][j]
		}
		sum += l.biases[j]
		outputs[j] = sum
	}

	// For training with mini-batch we usually use ForwardBatch, but for a single sample, skip batch norm.
	if !training {
		for j := 0; j < l.outputs; j++ {
			outputs[j] = l.gamma[j]*outputs[j] + l.beta[j]
		}
	}

	// Apply activation function.
	for j := 0; j < l.outputs; j++ {
		switch l.activation {
		case SigmoidActivation:
			outputs[j] = sigmoid(outputs[j])
		case ReLUActivation:
			outputs[j] = relu(outputs[j])
		case TanhActivation:
			outputs[j] = tanh(outputs[j])
		case LeakyReLUActivation:
			outputs[j] = leakyReLU(outputs[j])
		case SoftmaxActivation:
			// Softmax should be applied to the whole vector later.
		case IdentityActivation:
			// Identity: do nothing.
		}
	}

	// For consistency, returning nil for batch stats.
	return outputs, nil, nil
}

// ForwardBatch processes a batch of samples through the layer.
// inputs is a slice of samples, each sample being a slice of float64.
func (l *Layer) ForwardBatch(inputs [][]float64, training bool) ([][]float64, []float64, []float64) {
	batchSize := len(inputs)
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
	// If training and the layer is not using Identity activation, apply batch normalization.
	if training && l.activation != IdentityActivation {
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
			batchVariances[j] = sumSq / float64(batchSize)
		}
		// Normalize outputs using the computed batch statistics.
		for b := 0; b < batchSize; b++ {
			for j := 0; j < l.outputs; j++ {
				outputs[b][j] = l.gamma[j]*(outputs[b][j]-batchMeans[j])/math.Sqrt(batchVariances[j]+1e-8) + l.beta[j]
			}
		}
	} else {
		// If not training or if IdentityActivation is used, simply apply gamma and beta.
		for b := 0; b < batchSize; b++ {
			for j := 0; j < l.outputs; j++ {
				outputs[b][j] = l.gamma[j]*outputs[b][j] + l.beta[j]
			}
		}
	}

	// Apply activation function to each output element.
	for b := 0; b < batchSize; b++ {
		for j := 0; j < l.outputs; j++ {
			switch l.activation {
			case SigmoidActivation:
				outputs[b][j] = sigmoid(outputs[b][j])
			case ReLUActivation:
				outputs[b][j] = relu(outputs[b][j])
			case TanhActivation:
				outputs[b][j] = tanh(outputs[b][j])
			case LeakyReLUActivation:
				outputs[b][j] = leakyReLU(outputs[b][j])
			case SoftmaxActivation:
				// Softmax will be applied to the entire vector later.
			case IdentityActivation:
				// For identity activation, leave the output as is.
			}
		}
	}

	return outputs, batchMeans, batchVariances
}

// calculateError computes mean squared error for a given sample.
func calculateError(outputs []float64, targets []float64) float64 {
	error := 0.0
	for i := range outputs {
		error += 0.5 * math.Pow(targets[i]-outputs[i], 2)
	}
	return error
}

// NeuralNetwork structure
type NeuralNetwork struct {
	layers           []*Layer
	learningRate     float64
	l2Regularization float64
}

// NewNeuralNetwork initializes a neural network.
// layerSizes is a slice of integers indicating the number of neurons in each layer.
// activations is a slice of activation types for each layer except the input layer.
func NewNeuralNetwork(layerSizes []int, activations []int, learningRate float64, l2Regularization float64) *NeuralNetwork {
	nn := &NeuralNetwork{learningRate: learningRate, l2Regularization: l2Regularization}
	for i := 0; i < len(layerSizes)-1; i++ {
		nn.layers = append(nn.layers, NewLayer(layerSizes[i], layerSizes[i+1], activations[i]))
	}
	return nn
}

// PredictRegression performs a forward pass on a single sample (without softmax on the output layer).
// This method is exported for inference.
func (nn *NeuralNetwork) PredictRegression(input []float64) []float64 {
	currentOutput := input
	for _, layer := range nn.layers {
		out, _, _ := layer.Forward(currentOutput, false)
		currentOutput = out
	}
	return currentOutput
}

// Train trains the neural network using mini-batch gradient descent.
// It logs the average training loss every printEvery iterations.
func (nn *NeuralNetwork) Train(inputs [][]float64, targets [][]float64, iterations int,
	learningRateDecayFactor float64, decayEpochs int, miniBatchSize int) {

	totalSamples := len(inputs)
	printEvery := 100000 // adjust as desired

	startTime := time.Now()

	for iter := 0; iter < iterations; iter++ {
		// Decay the learning rate periodically.
		if iter > 0 && iter%decayEpochs == 0 {
			nn.learningRate *= learningRateDecayFactor
		}

		// Process mini-batches.
		for start := 0; start < totalSamples; start += miniBatchSize {
			end := start + miniBatchSize
			if end > totalSamples {
				end = totalSamples
			}
			batchInputs := inputs[start:end]
			batchTargets := targets[start:end]
			batchSize := len(batchInputs)

			// Forward pass through all layers.
			// outputs[k] will be a batch ([][]float64) for layer k, with outputs[0] being the batchInputs.
			outputs := make([][][]float64, len(nn.layers)+1)
			outputs[0] = batchInputs

			// Store batch normalization statistics if needed.
			batchMeans := make([][]float64, len(nn.layers))
			batchVariances := make([][]float64, len(nn.layers))

			for k := 1; k <= len(nn.layers); k++ {
				out, means, variances := nn.layers[k-1].ForwardBatch(outputs[k-1], true)
				outputs[k] = out
				batchMeans[k-1] = means
				batchVariances[k-1] = variances
			}

			finalOutputs := outputs[len(nn.layers)] // shape: batchSize x outputDimension

			// Compute error for the output layer.
			errors := make([][][]float64, len(nn.layers))
			errors[len(nn.layers)-1] = make([][]float64, batchSize)
			for b := 0; b < batchSize; b++ {
				errVec := make([]float64, len(finalOutputs[b]))
				for j := 0; j < len(finalOutputs[b]); j++ {
					errVec[j] = batchTargets[b][j] - finalOutputs[b][j]
				}
				errors[len(nn.layers)-1][b] = errVec
			}

			// Backward pass: loop from last layer to first.
			for l := len(nn.layers) - 1; l >= 0; l-- {
				// For each neuron j in layer l, compute the average gradient over the mini-batch.
				for j := 0; j < nn.layers[l].outputs; j++ {
					gradientSum := 0.0
					for b := 0; b < batchSize; b++ {
						var grad float64
						// Compute derivative based on the activation.
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
						case IdentityActivation:
							grad = 1.0
						}
						gradientSum += errors[l][b][j] * grad
					}
					avgGradient := gradientSum / float64(batchSize)
					// Update weights for neuron j.
					for i := 0; i < nn.layers[l].inputs; i++ {
						deltaSum := 0.0
						for b := 0; b < batchSize; b++ {
							deltaSum += avgGradient * outputs[l][b][i]
						}
						avgDelta := (deltaSum / float64(batchSize)) * nn.learningRate
						nn.layers[l].weights[i][j] += avgDelta
						nn.layers[l].weights[i][j] -= nn.l2Regularization * nn.learningRate * nn.layers[l].weights[i][j]
					}
					// Update biases and batch norm parameters.
					nn.layers[l].biases[j] += avgGradient * nn.learningRate
					nn.layers[l].gamma[j] += avgGradient * nn.learningRate
					nn.layers[l].beta[j] += avgGradient * nn.learningRate
				}

				// Propagate errors to the previous layer.
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
			} // end backward pass
		} // end mini-batch loop

		// Optionally log training loss every printEvery iterations.
		if iter%printEvery == 0 {
			totalLoss := 0.0
			sampleCount := 0
			// Compute loss over entire training set.
			for b := 0; b < totalSamples; b++ {
				// Forward pass for each sample.
				current := inputs[b]
				for _, layer := range nn.layers {
					out, _, _ := layer.Forward(current, false)
					current = out
				}
				loss := calculateError(current, targets[b])
				totalLoss += loss
				sampleCount++
			}
			avgLoss := totalLoss / float64(sampleCount)
			fmt.Printf("Iteration %d, Average Loss: %v\n", iter, avgLoss)
		}
	} // end iterations

	fmt.Printf("Training completed in %v\n", time.Since(startTime))
}
