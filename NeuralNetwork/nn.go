package neuralnetwork

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// NeuralNetwork: fully-connected feedforward network with backprop, SGD, early stopping, and LR scheduling.
type NeuralNetwork struct {
	LayerSizes      []int // sizes for each layer
	NumLayers       int   // number of layers
	Weights         [][][]float64
	Biases          [][]float64
	LearningRate    float64
	Epochs          int
	BatchSize       int
	InitialPatience int // initial early stopping patience
	patienceCounter int // countdown
	BestValLoss     float64
	TrainLosses     []float64 // record of training losses
	ValLosses       []float64 // record of validation losses
}

// NewNetwork: initialize weights & biases with Xavier init, set patience
func NewNetwork(sizes []int, lr float64, epochs, batchSize, patience int) *NeuralNetwork {
	nn := &NeuralNetwork{
		LayerSizes:      append([]int(nil), sizes...),
		NumLayers:       len(sizes),
		LearningRate:    lr,
		Epochs:          epochs,
		BatchSize:       batchSize,
		InitialPatience: patience,
		BestValLoss:     math.Inf(1),
		TrainLosses:     make([]float64, 0, epochs),
		ValLosses:       make([]float64, 0, epochs),
	}
	// initialize counters
	nn.patienceCounter = patience

	nn.Weights = make([][][]float64, nn.NumLayers-1)
	nn.Biases = make([][]float64, nn.NumLayers-1)
	// Xavier initialization
	for l := 0; l < nn.NumLayers-1; l++ {
		in, out := sizes[l], sizes[l+1]
		limit := math.Sqrt(6.0 / float64(in+out))
		nn.Weights[l] = make([][]float64, out)
		for i := 0; i < out; i++ {
			nn.Weights[l][i] = make([]float64, in)
			for j := 0; j < in; j++ {
				nn.Weights[l][i][j] = rand.Float64()*2*limit - limit
			}
		}
		// biases init to zero
		nn.Biases[l] = make([]float64, out)
	}
	return nn
}

// Activation and derivative
func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }
func sigmoidPrime(x float64) float64 {
	s := sigmoid(x)
	return s * (1 - s)
}

// forward: returns activations and pre-activations
func (nn *NeuralNetwork) forward(x []float64) (acts [][]float64, zs [][]float64) {
	acts = make([][]float64, nn.NumLayers)
	zs = make([][]float64, nn.NumLayers-1)
	acts[0] = append([]float64(nil), x...)
	for l := 0; l < nn.NumLayers-1; l++ {
		inAct := acts[l]
		nOut := nn.LayerSizes[l+1]
		z := make([]float64, nOut)
		a := make([]float64, nOut)
		for i := 0; i < nOut; i++ {
			z[i] = nn.Biases[l][i]
			for j, v := range inAct {
				z[i] += nn.Weights[l][i][j] * v
			}
			// identity activation on output layer
			if l < nn.NumLayers-2 {
				a[i] = sigmoid(z[i])
			} else {
				a[i] = z[i]
			}
		}
		zs[l] = z
		acts[l+1] = a
	}
	return
}

// backprop: returns gradients
func (nn *NeuralNetwork) backprop(x, y []float64) ([][][]float64, [][]float64) {
	acts, zs := nn.forward(x)
	nablaW := make([][][]float64, nn.NumLayers-1)
	nablaB := make([][]float64, nn.NumLayers-1)
	for l := range nablaW {
		nOut := nn.LayerSizes[l+1]
		nablaW[l] = make([][]float64, nOut)
		nablaB[l] = make([]float64, nOut)
		for i := 0; i < nOut; i++ {
			nablaW[l][i] = make([]float64, nn.LayerSizes[l])
		}
	}
	// output layer error (identity derivative = 1)
	L := nn.NumLayers - 1
	delta := make([]float64, nn.LayerSizes[L])
	for i := range delta {
		delta[i] = acts[L][i] - y[i]
		nablaB[L-1][i] = delta[i]
		for j := range acts[L-1] {
			nablaW[L-1][i][j] = delta[i] * acts[L-1][j]
		}
	}
	// backpropagate
	for l := L - 2; l >= 0; l-- {
		newDelta := make([]float64, nn.LayerSizes[l+1])
		for i := 0; i < nn.LayerSizes[l+1]; i++ {
			sum := 0.0
			for j := 0; j < nn.LayerSizes[l+2]; j++ {
				sum += nn.Weights[l+1][j][i] * delta[j]
			}
			newDelta[i] = sum * sigmoidPrime(zs[l][i])
			nablaB[l][i] = newDelta[i]
			for j := range acts[l] {
				nablaW[l][i][j] = newDelta[i] * acts[l][j]
			}
		}
		delta = newDelta
	}
	return nablaW, nablaB
}

// updateBatch: apply SGD on one batch
func (nn *NeuralNetwork) updateBatch(batchX, batchY [][]float64) {
	m := float64(len(batchX))
	nablaWSum := make([][][]float64, len(nn.Weights))
	nablaBSum := make([][]float64, len(nn.Biases))
	for l := range nn.Weights {
		nOut := nn.LayerSizes[l+1]
		nablaWSum[l] = make([][]float64, nOut)
		nablaBSum[l] = make([]float64, nOut)
		for i := 0; i < nOut; i++ {
			nablaWSum[l][i] = make([]float64, nn.LayerSizes[l])
		}
	}
	for i := range batchX {
		nw, nb := nn.backprop(batchX[i], batchY[i])
		for l := range nw {
			for j := range nw[l] {
				for k := range nw[l][j] {
					nablaWSum[l][j][k] += nw[l][j][k]
				}
				nablaBSum[l][j] += nb[l][j]
			}
		}
	}
	for l := range nn.Weights {
		for i := range nn.Weights[l] {
			for j := range nn.Weights[l][i] {
				nn.Weights[l][i][j] -= (nn.LearningRate / m) * nablaWSum[l][i][j]
			}
			nn.Biases[l][i] -= (nn.LearningRate / m) * nablaBSum[l][i]
		}
	}
}

// MeanSquaredError
func MeanSquaredError(preds, targets [][]float64) float64 {
	total := 0.0
	for i := range preds {
		for j := range preds[i] {
			d := preds[i][j] - targets[i][j]
			total += d * d
		}
	}
	return total / float64(len(preds))
}

// Train: with early stopping and logging of train/val loss
func (nn *NeuralNetwork) Train(trainX, trainY, valX, valY [][]float64) {
	nn.patienceCounter = nn.InitialPatience
	for epoch := 1; epoch <= nn.Epochs; epoch++ {
		// shuffle & mini-batches
		perm := rand.Perm(len(trainX))
		for i := 0; i < len(trainX); i += nn.BatchSize {
			end := i + nn.BatchSize
			if end > len(trainX) {
				end = len(trainX)
			}
			batchX, batchY := [][]float64{}, [][]float64{}
			for _, idx := range perm[i:end] {
				batchX = append(batchX, trainX[idx])
				batchY = append(batchY, trainY[idx])
			}
			nn.updateBatch(batchX, batchY)
		}
		// compute train loss
		trainPreds := make([][]float64, len(trainX))
		for i, x := range trainX {
			trainPreds[i] = nn.Predict(x)
		}
		trainLoss := MeanSquaredError(trainPreds, trainY)
		// compute val loss
		valPreds := make([][]float64, len(valX))
		for i, x := range valX {
			valPreds[i] = nn.Predict(x)
		}
		valLoss := MeanSquaredError(valPreds, valY)

		// record
		nn.TrainLosses = append(nn.TrainLosses, trainLoss)
		nn.ValLosses = append(nn.ValLosses, valLoss)
		fmt.Printf("Epoch %d: train loss=%.6f, val loss=%.6f\n", epoch, trainLoss, valLoss)

		// early stopping
		if valLoss < nn.BestValLoss {
			nn.BestValLoss = valLoss
			nn.patienceCounter = nn.InitialPatience
		} else {
			nn.patienceCounter--
			if nn.patienceCounter <= 0 {
				fmt.Println("Early stopping.")
				return
			}
		}
	}
}

// Predict single sample
func (nn *NeuralNetwork) Predict(x []float64) []float64 {
	acts, _ := nn.forward(x)
	return acts[nn.NumLayers-1]
}

// Dataset generation for arithmetic sequences
func generateArithData(nSamples, seqLen, predLen int) ([][]float64, [][]float64) {
	inputs := make([][]float64, nSamples)
	targets := make([][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		start := rand.Float64()*10 - 5
		step := rand.Float64()*4 - 2
		seq := make([]float64, seqLen+predLen)
		for j := range seq {
			seq[j] = start + float64(j)*step
		}
		inputs[i] = makeSeqFeatures(seq[:seqLen])
		targets[i] = seq[seqLen:]
	}
	return inputs, targets
}

func makeSeqFeatures(x []float64) []float64 {
	n := len(x)
	feat := make([]float64, 0, 4*n-1)
	// raw
	feat = append(feat, x...)
	// diffs
	for i := 1; i < n; i++ {
		feat = append(feat, x[i]-x[i-1])
	}
	// positions
	for i := 0; i < n; i++ {
		feat = append(feat, float64(i+1))
	}
	// squares
	for _, v := range x {
		feat = append(feat, v*v)
	}
	return feat
}

// normalization (z-score)
func FitScaler(data [][]float64) (means, stds []float64) {
	nFeatures := len(data[0])
	means = make([]float64, nFeatures)
	stds = make([]float64, nFeatures)
	for j := 0; j < nFeatures; j++ {
		for i := range data {
			means[j] += data[i][j]
		}
		means[j] /= float64(len(data))
		for i := range data {
			d := data[i][j] - means[j]
			stds[j] += d * d
		}
		stds[j] = math.Sqrt(stds[j] / float64(len(data)))
		if stds[j] == 0 {
			stds[j] = 1
		}
	}
	return
}

func Normalize(data [][]float64, means, stds []float64) [][]float64 {
	norm := make([][]float64, len(data))
	for i := range data {
		norm[i] = make([]float64, len(data[i]))
		for j := range data[i] {
			norm[i][j] = (data[i][j] - means[j]) / stds[j]
		}
	}
	return norm
}

func Denormalize(vec, means, stds []float64) []float64 {
	down := make([]float64, len(vec))
	for i := range vec {
		down[i] = vec[i]*stds[i] + means[i]
	}
	return down
}

// EvaluateMetrics calculates MSE, RMSE, MAE, and R² for predictions vs. targets
func EvaluateMetrics(preds, targets [][]float64) {
	n := len(preds)
	if n == 0 || len(preds[0]) == 0 {
		fmt.Println("Empty predictions or targets")
		return
	}

	mse := 0.0
	mae := 0.0
	ssRes := 0.0
	ssTot := 0.0
	mean := 0.0

	// Calculate mean of target values
	for i := range targets {
		for j := range targets[i] {
			mean += targets[i][j]
		}
	}
	mean /= float64(n * len(targets[0]))

	// Compute metrics
	for i := range preds {
		for j := range preds[i] {
			diff := preds[i][j] - targets[i][j]
			mse += diff * diff
			mae += math.Abs(diff)

			ssRes += diff * diff
			ssTot += math.Pow(targets[i][j]-mean, 2)
		}
	}

	mse /= float64(n)
	rmse := math.Sqrt(mse)
	mae /= float64(n)
	r2 := 1 - ssRes/ssTot

	fmt.Printf("Evaluation Metrics:\n")
	fmt.Printf("MSE  = %.6f\n", mse)
	fmt.Printf("RMSE = %.6f\n", rmse)
	fmt.Printf("MAE  = %.6f\n", mae)
	fmt.Printf("R²   = %.6f\n", r2)
}
