// mini_transformer.go - Regenerated clean solution
package gpt

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// Vocabulary setup
var vocab = []string{"i", "love", "learning", "python", "is", "fun", "enjoy", "new", "things", "to", "code", "every", "day", "never", "stops", "am", "golang", "and", "go", "are", "awesome"}
var wordToIdx = map[string]int{}
var idxToWord = map[int]string{}
var vocabSize = len(vocab)

// Hyperparameters
const (
	embedDim     = 4
	seqLen       = 3
	numHeads     = 2
	headDim      = embedDim / numHeads
	learningRate = 0.01
	epochs       = 100
)

// Model Parameters
var (
	embedding  [][]float64
	Wff1       [][]float64
	Bff1       []float64
	Wff2       [][]float64
	Bff2       []float64
	Wq, Wk, Wv [][][]float64
	Wout       [][]float64
	Wo         [][]float64
	bOut       []float64
	posEnc     [][]float64
)

// Initialization
func init() {
	for i, w := range vocab {
		wordToIdx[w] = i
		idxToWord[i] = w
	}

	embedding = randMatrix(vocabSize, embedDim)
	Wq, Wk, Wv = make([][][]float64, numHeads), make([][][]float64, numHeads), make([][][]float64, numHeads)
	for i := 0; i < numHeads; i++ {
		Wq[i] = randMatrix(embedDim, headDim)
		Wk[i] = randMatrix(embedDim, headDim)
		Wv[i] = randMatrix(embedDim, headDim)
	}

	Wout = randMatrix(embedDim, embedDim)
	Wo = randMatrix(vocabSize, embedDim)
	bOut = make([]float64, vocabSize)
	posEnc = positionalEncoding(seqLen, embedDim)
	Wff1 = randMatrix(embedDim, embedDim)
	Bff1 = make([]float64, embedDim)
	Wff2 = randMatrix(embedDim, embedDim)
	Bff2 = make([]float64, embedDim)
}

// Training data
var trainingData = [][]string{
	{"i", "love", "learning", "python"},
	{"python", "is", "fun"},
	{"i", "enjoy", "learning"},
	{"love", "learning", "new", "things"},
	{"i", "love", "to", "code"},
	{"code", "every", "day"},
	{"learning", "never", "stops"},
	{"i", "am", "learning", "golang"},
	{"python", "and", "go", "are", "awesome"},
}

// Utility functions
func randMatrix(rows, cols int) [][]float64 {
	mat := make([][]float64, rows)
	for i := range mat {
		mat[i] = make([]float64, cols)
		for j := range mat[i] {
			mat[i][j] = rand.NormFloat64()
		}
	}
	return mat
}

func positionalEncoding(seqLen, embedDim int) [][]float64 {
	encoding := make([][]float64, seqLen)
	for pos := 0; pos < seqLen; pos++ {
		encoding[pos] = make([]float64, embedDim)
		for i := 0; i < embedDim; i++ {
			angleRate := float64(pos) / math.Pow(10000, float64(2*(i/2))/float64(embedDim))
			if i%2 == 0 {
				encoding[pos][i] = math.Sin(angleRate)
			} else {
				encoding[pos][i] = math.Cos(angleRate)
			}
		}
	}
	return encoding
}

func softmax(x []float64) []float64 {
	maxVal := x[0]
	for _, v := range x {
		if v > maxVal {
			maxVal = v
		}
	}
	expSum := 0.0
	expVals := make([]float64, len(x))
	for i, v := range x {
		expVals[i] = math.Exp(v - maxVal)
		expSum += expVals[i]
	}
	for i := range expVals {
		expVals[i] /= expSum
	}
	return expVals
}

func dot(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func matVecMul(mat [][]float64, vec []float64) []float64 {
	out := make([]float64, len(mat))
	for i := range mat {
		out[i] = dot(mat[i], vec)
	}
	return out
}

func matMatMul(a, b [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i := range a {
		out[i] = make([]float64, len(b[0]))
		for j := range b[0] {
			for k := range a[0] {
				out[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return out
}

func project(x [][]float64, W [][]float64) [][]float64 {
	out := make([][]float64, len(x))
	for i, row := range x {
		out[i] = matVecMul(W, row)
	}
	return out
}

func scaledDotProduct(Q, K [][]float64) [][]float64 {
	n := len(Q)
	scores := make([][]float64, n)
	for i := range scores {
		scores[i] = make([]float64, n)
		for j := range scores[i] {
			scores[i][j] = dot(Q[i], K[j]) / math.Sqrt(float64(len(Q[0])))
		}
	}
	return scores
}

func softmax2D(x [][]float64) [][]float64 {
	out := make([][]float64, len(x))
	for i := range x {
		out[i] = softmax(x[i])
	}
	return out
}

func flatten(mat [][]float64) []float64 {
	out := []float64{}
	for _, row := range mat {
		out = append(out, row...)
	}
	return out
}

func unflatten(flat [][]float64) [][]float64 {
	return flat
}

func layerNorm(x []float64) []float64 {
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(len(x))

	varSum := 0.0
	for _, v := range x {
		varSum += (v - mean) * (v - mean)
	}
	varMean := varSum / float64(len(x))
	std := math.Sqrt(varMean + 1e-5)

	norm := make([]float64, len(x))
	for i := range x {
		norm[i] = (x[i] - mean) / std
	}
	return norm
}

func addVec(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func meanVecs(vecs [][]float64) []float64 {
	return layerNorm(meanRaw(vecs))
}

func meanRaw(vecs [][]float64) []float64 {
	n := len(vecs)
	out := make([]float64, len(vecs[0]))
	for _, vec := range vecs {
		for i := range vec {
			out[i] += vec[i]
		}
	}
	for i := range out {
		out[i] /= float64(n)
	}
	return out
}

func crossEntropy(probs []float64, target int) float64 {
	return -math.Log(probs[target] + 1e-9)
}

func multiHeadAttention(x [][]float64) [][]float64 {
	heads := make([][]float64, numHeads)
	for h := 0; h < numHeads; h++ {
		Q := project(x, Wq[h])
		K := project(x, Wk[h])
		V := project(x, Wv[h])
		scores := scaledDotProduct(Q, K)
		weights := softmax2D(scores)
		head := matMatMul(weights, V)
		heads[h] = flatten(head)
	}
	return unflatten(heads)
}

func predictNextToken(inputIndices []int, temperature float64, training bool) ([]float64, []float64) {
	const dropoutRate = 0.1
	x := make([][]float64, len(inputIndices))
	for i, idx := range inputIndices {
		x[i] = addVec(embedding[idx], posEnc[i])
	}
	context := multiHeadAttention(x)
	summary := meanVecs(context)

	hidden := matVecMul(Wff1, summary)
	for i := range hidden {
		hidden[i] += Bff1[i]
		if hidden[i] < 0 {
			hidden[i] = 0
			if training && rand.Float64() < dropoutRate {
				hidden[i] = 0
			}
		}
	}

	projected := matVecMul(Wff2, hidden)
	for i := range projected {
		projected[i] += Bff2[i]
	}

	logits := make([]float64, vocabSize)
	for i := range logits {
		logits[i] = dot(Wo[i], projected) + bOut[i]
		logits[i] /= temperature
	}
	return softmax(logits), projected
}

func Train() {
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for _, seq := range trainingData {
			if len(seq) <= seqLen {
				continue
			}
			for i := 0; i <= len(seq)-seqLen-1; i++ {
				inputSeq := seq[i : i+seqLen]
				target := wordToIdx[seq[i+seqLen]]
				inputIndices := make([]int, seqLen)
				for j, word := range inputSeq {
					inputIndices[j] = wordToIdx[word]
				}
				probs, summary := predictNextToken(inputIndices, 0.7, true)
				loss := crossEntropy(probs, target)
				totalLoss += loss

				dlogits := make([]float64, len(probs))
				copy(dlogits, probs)
				dlogits[target] -= 1
				for i := range dlogits {
					if dlogits[i] > 1.0 {
						dlogits[i] = 1.0
					} else if dlogits[i] < -1.0 {
						dlogits[i] = -1.0
					}
				}

				for i := 0; i < len(Wo); i++ {
					for j := 0; j < len(Wo[i]); j++ {
						if j < len(summary) {
							Wo[i][j] -= learningRate * summary[j] * dlogits[i]
						}
					}
				}
				for i := range bOut {
					bOut[i] -= learningRate * dlogits[i]
				}
			}
		}
		if epoch%10 == 0 {
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss)
		}
	}
}

func GenerateText(start []string, tokens int) string {
	seq := append([]string{}, start...)
	tempStart := 1.0
	tempEnd := 0.5

	for i := 0; i < tokens; i++ {
		temp := tempStart - (tempStart-tempEnd)*float64(i)/float64(tokens-1)
		var input []int
		if len(seq) < seqLen {
			padding := make([]int, seqLen-len(seq))
			for j := range padding {
				padding[j] = 0
			}
			for _, word := range seq {
				input = append(input, wordToIdx[word])
			}
			input = append(padding, input...)
		} else {
			for _, word := range seq[len(seq)-seqLen:] {
				input = append(input, wordToIdx[word])
			}
		}
		probs, _ := predictNextToken(input, temp, false)
		maxIdx := sample(probs)
		seq = append(seq, idxToWord[maxIdx])
	}
	return strings.Join(seq, " ")
}

func sample(probs []float64) int {
	r := rand.Float64()
	cumulative := 0.0
	for i, p := range probs {
		cumulative += p
		if r < cumulative {
			return i
		}
	}
	return len(probs) - 1
}
