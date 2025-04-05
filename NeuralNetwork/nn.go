package NeuralNetwork

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/infiniteCrank/mathbot/db"
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

// Adam hyperparameters (tunable)
const (
	beta1   = 0.9
	beta2   = 0.999
	epsilon = 1e-8
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

	// Adam optimizer parameters for weights and biases:
	weightM [][]float64 // First moment estimates for weights
	weightV [][]float64 // Second moment estimates for weights
	biasM   []float64   // First moment estimates for biases
	biasV   []float64   // Second moment estimates for biases
}

// NewLayer creates and initializes a new layer.
func NewLayer(inputs, outputs int, activation int) *Layer {
	weights := make([][]float64, inputs)
	weightM := make([][]float64, inputs)
	weightV := make([][]float64, inputs)
	for i := range weights {
		weights[i] = make([]float64, outputs)
		weightM[i] = make([]float64, outputs)
		weightV[i] = make([]float64, outputs)
		for j := range weights[i] {
			// Scale down the initialization to help training stability.
			weights[i][j] = rand.NormFloat64() * 0.01
			weightM[i][j] = 0.0
			weightV[i][j] = 0.0
		}
	}

	biases := make([]float64, outputs)
	biasM := make([]float64, outputs)
	biasV := make([]float64, outputs)
	for j := range biases {
		biases[j] = 0.0
		biasM[j] = 0.0
		biasV[j] = 0.0
	}

	gamma := make([]float64, outputs)
	beta := make([]float64, outputs)
	for j := range gamma {
		gamma[j] = 1.0
		beta[j] = 0.0
	}

	return &Layer{
		inputs:     inputs,
		outputs:    outputs,
		weights:    weights,
		biases:     biases,
		activation: activation,
		gamma:      gamma,
		beta:       beta,
		weightM:    weightM,
		weightV:    weightV,
		biasM:      biasM,
		biasV:      biasV,
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
	Layers           []*Layer
	learningRate     float64
	l2Regularization float64
}

// NewNeuralNetwork initializes a neural network.
// layerSizes is a slice of integers indicating the number of neurons in each layer.
// activations is a slice of activation types for each layer except the input layer.
func NewNeuralNetwork(layerSizes []int, activations []int, learningRate float64, l2Regularization float64) *NeuralNetwork {
	nn := &NeuralNetwork{learningRate: learningRate, l2Regularization: l2Regularization}
	for i := 0; i < len(layerSizes)-1; i++ {
		nn.Layers = append(nn.Layers, NewLayer(layerSizes[i], layerSizes[i+1], activations[i]))
	}
	return nn
}

// PredictRegression performs a forward pass on a single sample (without softmax on the output layer).
// This method is exported for inference.
func (nn *NeuralNetwork) PredictRegression(input []float64) []float64 {
	currentOutput := input
	for _, layer := range nn.Layers {
		out, _, _ := layer.Forward(currentOutput, false)
		currentOutput = out
	}
	return currentOutput
}

// Train trains the neural network using mini-batch gradient descent with Adam optimizer.
// It logs the average training loss every printEvery iterations and implements a patience mechanism for early stopping.
func (nn *NeuralNetwork) Train(inputs [][]float64, targets [][]float64, iterations int,
	learningRateDecayFactor float64, decayEpochs int, miniBatchSize int) {

	totalSamples := len(inputs)
	printEvery := 100 // Adjust as desired

	// Patience mechanism variables:
	bestLoss := math.MaxFloat64
	patienceCounter := 0
	maxPatience := 500 // Number of iterations with no improvement allowed before stopping

	startTime := time.Now()
	adamTimeStep := 0 // Global counter for Adam updates.

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
			outputs := make([][][]float64, len(nn.Layers)+1)
			outputs[0] = batchInputs

			for k := 1; k <= len(nn.Layers); k++ {
				out, _, _ := nn.Layers[k-1].ForwardBatch(outputs[k-1], true)
				outputs[k] = out
			}

			finalOutputs := outputs[len(nn.Layers)] // shape: batchSize x outputDimension

			// Compute error for the output layer.
			errors := make([][][]float64, len(nn.Layers))
			errors[len(nn.Layers)-1] = make([][]float64, batchSize)
			for b := 0; b < batchSize; b++ {
				errVec := make([]float64, len(finalOutputs[b]))
				for j := 0; j < len(finalOutputs[b]); j++ {
					errVec[j] = batchTargets[b][j] - finalOutputs[b][j]
				}
				errors[len(nn.Layers)-1][b] = errVec
			}

			// Increment global Adam time step.
			adamTimeStep++

			// Backward pass: loop from last layer to first.
			for l := len(nn.Layers) - 1; l >= 0; l-- {
				for j := 0; j < nn.Layers[l].outputs; j++ {
					gradientSum := 0.0
					for b := 0; b < batchSize; b++ {
						var grad float64
						switch nn.Layers[l].activation {
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
					// Update weights for neuron j using Adam.
					for i := 0; i < nn.Layers[l].inputs; i++ {
						deltaSum := 0.0
						for b := 0; b < batchSize; b++ {
							deltaSum += avgGradient * outputs[l][b][i]
						}
						avgDelta := deltaSum / float64(batchSize)

						// Adam update for weight.
						m := beta1*nn.Layers[l].weightM[i][j] + (1-beta1)*avgDelta
						v := beta2*nn.Layers[l].weightV[i][j] + (1-beta2)*avgDelta*avgDelta
						mHat := m / (1 - math.Pow(beta1, float64(adamTimeStep)))
						vHat := v / (1 - math.Pow(beta2, float64(adamTimeStep)))
						weightUpdate := nn.learningRate * mHat / (math.Sqrt(vHat) + epsilon)

						nn.Layers[l].weights[i][j] += weightUpdate

						// Apply L2 regularization.
						nn.Layers[l].weights[i][j] -= nn.l2Regularization * nn.learningRate * nn.Layers[l].weights[i][j]

						// Save updated moments.
						nn.Layers[l].weightM[i][j] = m
						nn.Layers[l].weightV[i][j] = v
					}

					// Adam update for bias.
					mBias := beta1*nn.Layers[l].biasM[j] + (1-beta1)*avgGradient
					vBias := beta2*nn.Layers[l].biasV[j] + (1-beta2)*avgGradient*avgGradient
					mHatBias := mBias / (1 - math.Pow(beta1, float64(adamTimeStep)))
					vHatBias := vBias / (1 - math.Pow(beta2, float64(adamTimeStep)))
					biasUpdate := nn.learningRate * mHatBias / (math.Sqrt(vHatBias) + epsilon)
					nn.Layers[l].biases[j] += biasUpdate

					// Save updated bias moments.
					nn.Layers[l].biasM[j] = mBias
					nn.Layers[l].biasV[j] = vBias
				}

				// Propagate errors to the previous layer.
				if l > 0 {
					errors[l-1] = make([][]float64, batchSize)
					for b := 0; b < batchSize; b++ {
						prevErr := make([]float64, nn.Layers[l-1].outputs)
						for i := 0; i < nn.Layers[l-1].outputs; i++ {
							sumError := 0.0
							for j := 0; j < nn.Layers[l].outputs; j++ {
								sumError += errors[l][b][j] * nn.Layers[l].weights[i][j]
							}
							prevErr[i] = sumError
						}
						errors[l-1][b] = prevErr
					}
				}
			} // End of backward pass
		} // End of mini-batch processing

		// Calculate average loss over the entire training set every printEvery iterations.
		if iter%printEvery == 0 {
			totalLoss := 0.0
			sampleCount := 0
			// Compute loss over entire training set.
			for b := 0; b < totalSamples; b++ {
				current := inputs[b]
				for _, layer := range nn.Layers {
					out, _, _ := layer.Forward(current, false)
					current = out
				}
				// Using MSE as loss measure (for a single output in regression)
				loss := CalculateMSE([]float64{current[0]}, []float64{targets[b][0]})
				totalLoss += loss
				sampleCount++
			}
			avgLoss := totalLoss / float64(sampleCount)
			fmt.Printf("Iteration %d, Average Loss: %v\n", iter, avgLoss)

			// Patience-based early stopping.
			if avgLoss < bestLoss {
				bestLoss = avgLoss
				patienceCounter = 0
			} else {
				patienceCounter++
			}
			if patienceCounter >= maxPatience {
				fmt.Printf("Early stopping at iteration %d: no improvement for %d iterations, best loss: %.5f\n", iter, maxPatience, bestLoss)
				break
			}
		}
	} // End of iterations

	fmt.Printf("Training completed in %v\n", time.Since(startTime))
}

func (nn *NeuralNetwork) SaveWeights() {
	// Connect to the database
	db := db.ConnectDB()

	for layerIndex, layer := range nn.Layers {
		for neuronIndex, weights := range layer.weights {
			for weightIndex, weight := range weights {
				_, err := db.ExecContext(context.Background(),
					"INSERT INTO nn_schema.nn_weights (layer_index, neuron_index, weight_index, weight) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
					layerIndex, neuronIndex, weightIndex, weight,
				)
				if err != nil {
					log.Printf("Error saving weight: %v", err)
				}
			}
		}
	}
}

func (nn *NeuralNetwork) LoadWeights() {
	// Connect to the database
	db := db.ConnectDB()

	// Initialize the database (create table if it doesn't exist).
	if err := InitDB(db); err != nil {
		log.Fatal("Error initializing database:", err)
	}

	rows, err := db.QueryContext(context.Background(), "SELECT layer_index, neuron_index, weight_index, weight FROM nn_schema.nn_weights")
	if err != nil {
		log.Fatalf("Error loading weights: %v", err)
	}
	defer rows.Close()

	for rows.Next() {
		var layerIndex, neuronIndex, weightIndex int
		var weight float64
		err := rows.Scan(&layerIndex, &neuronIndex, &weightIndex, &weight)
		if err != nil {
			log.Printf("Error scanning row: %v", err)
			continue
		}
		nn.Layers[layerIndex].weights[neuronIndex][weightIndex] = weight
	}
}

// InitDB creates the nn_weights table if it doesn't exist.
func InitDB(db *sql.DB) error {
	createTableQuery := `
		CREATE TABLE IF NOT EXISTS nn_schema.nn_weights (
			id SERIAL PRIMARY KEY,
			layer_index INT,
			neuron_index INT,
			weight_index INT,
			weight FLOAT8
		);
	`
	_, err := db.Exec(createTableQuery)
	return err
}

func (nn *NeuralNetwork) DeleteWeights() {
	// Connect to the database
	db := db.ConnectDB()
	defer db.Close() // Ensure the database connection is closed

	_, err := db.ExecContext(context.Background(),
		"DELETE FROM nn_schema.nn_weights")
	if err != nil {
		log.Printf("Error deleting weights: %v", err)
	}
}
