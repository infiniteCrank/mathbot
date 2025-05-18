// mini_transformer.go - Regenerated clean solution
package gpt

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
)

// Vocabulary setup
var vocab = []string{}
var wordToIdx = map[string]int{}
var idxToWord = map[int]string{}
var vocabSize = len(vocab)

// Hyperparameters
const (
	embedDim     = 64
	seqLen       = 16
	numHeads     = 2
	headDim      = embedDim / numHeads
	learningRate = 0.01
	epochs       = 100
)

// Model Parameters
var (
	embedding  [][]float64
	layers     []TransformerLayer
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
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func ExportVocabulary(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for i, token := range vocab {
		line := fmt.Sprintf("%d\t%s\n", i, token)
		_, err := f.WriteString(line)
		if err != nil {
			return err
		}
	}
	return nil
}

func Detokenize(tokens []int) string {
	var parts []string
	for _, idx := range tokens {
		if word, ok := idxToWord[idx]; ok {
			parts = append(parts, word)
		}
	}
	return strings.Join(parts, "")
}

func InitFromMarkdownFiles(files []string) {
	charMap := make(map[string]int)
	for _, text := range files {
		for _, char := range strings.Split(strings.ToLower(text), "") {
			if char != " " && char != "\n" && char != "	" {
				charMap[char]++
			}
		}
	}

	// Generate initial vocabulary
	vocab = []string{}
	for c := range charMap {
		vocab = append(vocab, c)
	}
	sort.Strings(vocab)

	// Byte Pair Encoding-like merge loop
	type pair struct{ A, B string }
	for mergeStep := 0; mergeStep < 100; mergeStep++ {
		pairFreq := make(map[pair]int)
		for _, text := range files {
			tokens := strings.Split(strings.ToLower(text), "")
			for i := 0; i < len(tokens)-1; i++ {
				a, b := tokens[i], tokens[i+1]
				if contains(vocab, a) && contains(vocab, b) {
					pairFreq[pair{a, b}]++
				}
			}
		}
		if len(pairFreq) == 0 {
			break
		}
		var maxPair pair
		maxCount := 0
		for p, c := range pairFreq {
			if c > maxCount {
				maxCount = c
				maxPair = p
			}
		}
		merged := maxPair.A + maxPair.B
		vocab = append(vocab, merged)
	}

	// Build vocab index
	wordToIdx = make(map[string]int)
	idxToWord = make(map[int]string)
	for i, c := range vocab {
		wordToIdx[c] = i
		idxToWord[i] = c
	}
	vocabSize = len(vocab)

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
	layers = make([]TransformerLayer, numLayers)
	for i := range layers {
		layers[i] = newTransformerLayer()
	}

	trainingData = [][]string{}
	for _, text := range files {
		text = strings.ToLower(text)
		sample := []string{}
		i := 0
		for i < len(text) {
			found := false
			for l := 4; l > 0; l-- {
				if i+l <= len(text) {
					sub := text[i : i+l]
					if _, ok := wordToIdx[sub]; ok {
						sample = append(sample, sub)
						i += l
						found = true
						break
					}
				}
			}
			if !found {
				i++
			}
		}
		if len(sample) <= seqLen {
			continue
		}
		for i := 0; i <= len(sample)-seqLen-1; i++ {
			window := sample[i : i+seqLen+1]
			trainingData = append(trainingData, window)
		}
	}

}

// Training data
var trainingData = [][]string{}

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
	d := float64(len(Q[0]))
	scores := make([][]float64, n)
	for i := range scores {
		scores[i] = make([]float64, n)
		for j := range scores[i] {
			if j > i {
				scores[i][j] = -1e9 // mask future positions (causal)
			} else {
				scores[i][j] = dot(Q[i], K[j]) / math.Sqrt(d)
			}
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

func transpose(mat [][]float64) [][]float64 {
	if len(mat) == 0 {
		return [][]float64{}
	}
	rows, cols := len(mat), len(mat[0])
	out := make([][]float64, cols)
	for i := range out {
		out[i] = make([]float64, rows)
		for j := 0; j < rows; j++ {
			out[i][j] = mat[j][i]
		}
	}
	return out
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

func multiHeadAttentionLayer(x [][]float64, l TransformerLayer) [][]float64 {
	Q := project(x, l.Wq)
	K := project(x, l.Wk)
	V := project(x, l.Wv)

	scores := scaledDotProduct(Q, K)
	for i := range scores {
		for j := range scores[i] {
			if j > i {
				scores[i][j] = math.Inf(-1) // causal mask
			}
		}
	}
	weights := softmax2D(scores)
	attention := matMatMul(weights, V)

	context := make([][]float64, len(attention))
	for i := range attention {
		context[i] = matVecMul(l.Wout, attention[i])
	}

	// Backpropagation for Wout
	// Assume dContext is the gradient from the next layer (placeholder for now)
	dContext := make([][]float64, len(context))
	for i := range dContext {
		dContext[i] = make([]float64, len(context[i]))
		for j := range dContext[i] {
			dContext[i][j] = 1.0 // placeholder gradient
		}
	}

	// Gradient of Wout: dL/dWout += dContext * attention^T
	for i := 0; i < len(l.Wout); i++ {
		for j := 0; j < len(l.Wout[i]); j++ {
			grad := 0.0
			for t := 0; t < len(attention); t++ {
				grad += dContext[t][i] * attention[t][j]
			}
			l.GradWout[i][j] += grad
		}
	}

	// Backpropagation for Q, K, V (simplified example)
	dAttention := make([][]float64, len(dContext))
	for t := 0; t < len(dContext); t++ {
		dAttention[t] = matVecMul(transpose(l.Wout), dContext[t])
	}

	// Backprop for Wv (dL/dV = softmax(scores)^T * dAttention)
	dV := matMatMul(transpose(softmax2D(scores)), dAttention)
	for i := range l.Wv {
		for j := range l.Wv[i] {
			grad := 0.0
			for t := range x {
				grad += dV[t][j] * x[t][i]
			}
			l.GradWv[i][j] += grad
		}
	}

	// Backpropagation for Wq and Wk using scores and dAttention

	dScores := make([][]float64, len(scores))
	for i := range scores {
		dScores[i] = make([]float64, len(scores[i]))
		for j := range scores[i] {
			if j <= i {
				for k := range dAttention[0] {
					dScores[i][j] += dAttention[i][k] * V[j][k]
				}
				dScores[i][j] /= math.Sqrt(float64(len(Q[0])))
			}
		}
	}

	// Gradient for Q
	for i := range l.Wq {
		for j := range l.Wq[i] {
			grad := 0.0
			for t := range Q {
				qGrad := 0.0
				for k := range K {
					if k <= t {
						qGrad += dScores[t][k] * K[k][j]
					}
				}
				grad += qGrad * x[t][i]
			}
			l.GradWq[i][j] += grad
		}
	}

	// Gradient for K
	for i := range l.Wk {
		for j := range l.Wk[i] {
			grad := 0.0
			for t := range K {
				kGrad := 0.0
				for k := range Q {
					if t <= k {
						kGrad += dScores[k][t] * Q[k][j]
					}
				}
				grad += kGrad * x[t][i]
			}
			l.GradWk[i][j] += grad
		}
	}

	return context
}

func zeroMatrix(rows, cols int) [][]float64 {
	z := make([][]float64, rows)
	for i := range z {
		z[i] = make([]float64, cols)
	}
	return z
}

func randMatrixLayer() TransformerLayer {
	return TransformerLayer{
		Wff1: randMatrix(embedDim, embedDim),
		Bff1: make([]float64, embedDim),
		Wff2: randMatrix(embedDim, embedDim),
		Bff2: make([]float64, embedDim),
		Wq:   randMatrix(embedDim, headDim),
		Wk:   randMatrix(embedDim, headDim),
		Wv:   randMatrix(embedDim, headDim),
		Wout: randMatrix(embedDim, embedDim),
	}
}

func adamUpdateLayer(l, m, v TransformerLayer, lr, beta1, beta2, eps float64, t int) TransformerLayer {
	update := func(w, m, v [][]float64) [][]float64 {
		out := make([][]float64, len(w))
		for i := range w {
			out[i] = make([]float64, len(w[i]))
			for j := range w[i] {
				mHat := m[i][j] / (1 - math.Pow(beta1, float64(t)))
				vHat := v[i][j] / (1 - math.Pow(beta2, float64(t)))
				out[i][j] = w[i][j] - lr*mHat/(math.Sqrt(vHat)+eps)
			}
		}
		return out
	}
	updateBias := func(b, m, v []float64) []float64 {
		out := make([]float64, len(b))
		for i := range b {
			mHat := m[i] / (1 - math.Pow(beta1, float64(t)))
			vHat := v[i] / (1 - math.Pow(beta2, float64(t)))
			out[i] = b[i] - lr*mHat/(math.Sqrt(vHat)+eps)
		}
		return out
	}
	l.Wff1 = update(l.Wff1, m.Wff1, v.Wff1)
	l.Bff1 = updateBias(l.Bff1, m.Bff1, v.Bff1)
	l.Wff2 = update(l.Wff2, m.Wff2, v.Wff2)
	l.Bff2 = updateBias(l.Bff2, m.Bff2, v.Bff2)
	l.Wq = update(l.Wq, m.Wq, v.Wq)
	l.Wk = update(l.Wk, m.Wk, v.Wk)
	l.Wv = update(l.Wv, m.Wv, v.Wv)
	l.Wout = update(l.Wout, m.Wout, v.Wout)
	return l
}

const numLayers = 2

type TransformerLayer struct {
	Wff1, GradWff1 [][]float64
	Bff1, GradBff1 []float64
	Wff2, GradWff2 [][]float64
	Bff2, GradBff2 []float64
	Wq, GradWq     [][]float64
	Wk, GradWk     [][]float64
	Wv, GradWv     [][]float64
	Wout, GradWout [][]float64
}

func reluDeriv(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func TransformerBlock(x []float64, l TransformerLayer, training bool, dropoutRate float64) []float64 {
	xNorm := layerNorm(x)
	ffnInput := layerNorm(xNorm) // positional layer norm before FFN

	// Forward pass
	hidden := make([]float64, len(l.Bff1))
	for i := range hidden {
		sum := 0.0
		for j := range ffnInput {
			sum += l.Wff1[i][j] * ffnInput[j]
		}
		sum += l.Bff1[i]
		if sum < 0 {
			sum = 0
			if training && rand.Float64() < dropoutRate {
				sum = 0
			}
		}
		hidden[i] = sum
	}

	projected := make([]float64, len(l.Bff2))
	for i := range projected {
		sum := 0.0
		for j := range hidden {
			sum += l.Wff2[i][j] * hidden[j]
		}
		projected[i] = sum + l.Bff2[i]
	}

	// Backpropagation for FFN layer (simplified example assuming L2 loss derivative available)
	dProject := make([]float64, len(projected))
	for i := range dProject {
		dProject[i] = projected[i] // this should come from next layer's gradient in real backprop
	}

	// Gradients for Wff2 and Bff2
	for i := range l.Wff2 {
		for j := range l.Wff2[i] {
			l.GradWff2[i][j] += dProject[i] * hidden[j]
		}
		l.GradBff2[i] += dProject[i]
	}

	// Backprop to hidden layer
	dHidden := make([]float64, len(hidden))
	for i := range l.Wff2[0] {
		sum := 0.0
		for j := range l.Wff2 {
			sum += l.Wff2[j][i] * dProject[j]
		}
		dHidden[i] = sum * reluDeriv(hidden[i])
	}

	// Gradients for Wff1 and Bff1
	for i := range l.Wff1 {
		for j := range l.Wff1[i] {
			l.GradWff1[i][j] += dHidden[i] * ffnInput[j]
		}
		l.GradBff1[i] += dHidden[i]
	}

	return addVec(projected, x) // residual connection
}

func predictNextToken(inputIndices []int, temperature float64, training bool) ([]float64, []float64) {
	const dropoutRate = 0.1
	x := make([][]float64, seqLen)
	pad := seqLen - len(inputIndices)
	for i := 0; i < pad; i++ {
		x[i] = addVec(embedding[0], posEnc[i]) // pad with "i" + position
	}
	for i := pad; i < seqLen; i++ {
		idx := inputIndices[i-pad]
		x[i] = addVec(embedding[idx], posEnc[i])
	}

	for layer := 0; layer < numLayers; layer++ {
		x = multiHeadAttentionLayer(x, layers[layer])
	}
	context := x

	summary := meanVecs(context)

	var xVec = summary
	for layer := 0; layer < numLayers; layer++ {
		xVec = TransformerBlock(xVec, layers[layer], training, dropoutRate)
	}

	logits := make([]float64, vocabSize)
	for i := range logits {
		logits[i] = dot(Wo[i], xVec) + bOut[i]
		logits[i] /= temperature
	}
	return softmax(logits), xVec
}

func Train() {
	const clipValue = 1.0
	beta1, beta2 := 0.9, 0.999
	eps := 1e-8

	mWo := make([][]float64, len(Wo))
	vWo := make([][]float64, len(Wo))
	for i := range Wo {
		mWo[i] = make([]float64, len(Wo[i]))
		vWo[i] = make([]float64, len(Wo[i]))
	}

	mbOut := make([]float64, len(bOut))
	vbOut := make([]float64, len(bOut))

	for epoch := 0; epoch < epochs; epoch++ {
		// Adam optimizers for each transformer layer
		mLayers := make([]TransformerLayer, len(layers))
		vLayers := make([]TransformerLayer, len(layers))
		for i := range layers {
			mLayers[i] = TransformerLayer{
				Wff1: randMatrix(embedDim, embedDim),
				Bff1: make([]float64, embedDim),
				Wff2: randMatrix(embedDim, embedDim),
				Bff2: make([]float64, embedDim),
				Wq:   randMatrix(embedDim, headDim),
				Wk:   randMatrix(embedDim, headDim),
				Wv:   randMatrix(embedDim, headDim),
				Wout: randMatrix(embedDim, embedDim),
			}
			vLayers[i] = randMatrixLayer()
		}
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
					if dlogits[i] > clipValue {
						dlogits[i] = clipValue
					} else if dlogits[i] < -clipValue {
						dlogits[i] = -clipValue
					}
				}

				for i := 0; i < len(Wo); i++ {
					for j := 0; j < len(Wo[i]); j++ {
						grad := summary[j] * dlogits[i]
						mWo[i][j] = beta1*mWo[i][j] + (1-beta1)*grad
						vWo[i][j] = beta2*vWo[i][j] + (1-beta2)*grad*grad
						mHat := mWo[i][j] / (1 - math.Pow(beta1, float64(epoch+1)))
						vHat := vWo[i][j] / (1 - math.Pow(beta2, float64(epoch+1)))
						Wo[i][j] -= learningRate * mHat / (math.Sqrt(vHat) + eps)
					}
				}
				for i := range bOut {
					grad := dlogits[i]
					mbOut[i] = beta1*mbOut[i] + (1-beta1)*grad
					vbOut[i] = beta2*vbOut[i] + (1-beta2)*grad*grad
					mHat := mbOut[i] / (1 - math.Pow(beta1, float64(epoch+1)))
					vHat := vbOut[i] / (1 - math.Pow(beta2, float64(epoch+1)))
					bOut[i] -= learningRate * mHat / (math.Sqrt(vHat) + eps)
				}
			}
		}
		if epoch%10 == 0 {
			fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, totalLoss)
		}

		// Update all transformer layer parameters using Adam
		for i := range layers {
			layers[i] = adamUpdateLayer(layers[i], mLayers[i], vLayers[i], learningRate, beta1, beta2, eps, epoch+1)
		}
	}

}

func GenerateText(start []string, tokens int) string {
	seq := append([]string{}, start...)
	tempStart := 1.0
	tempEnd := 0.5
	tokenHistory := []int{} // Track previously generated tokens for penalty

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
		applyRepetitionPenalty(probs, tokenHistory, 1.2)
		maxIdx := topKSample(probs, 5)
		seq = append(seq, idxToWord[maxIdx])
		tokenHistory = append(tokenHistory, maxIdx)
	}
	return strings.Join(seq, " ")
}

func applyRepetitionPenalty(probs []float64, history []int, penalty float64) {
	for _, idx := range history {
		probs[idx] /= penalty
	}
}

func topKSample(probs []float64, k int) int {
	type kv struct {
		idx int
		val float64
	}
	sorted := make([]kv, len(probs))
	for i, p := range probs {
		sorted[i] = kv{i, p}
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].val > sorted[j].val
	})

	topK := sorted[:k]
	sum := 0.0
	for _, item := range topK {
		sum += item.val
	}
	r := rand.Float64() * sum
	cumulative := 0.0
	for _, item := range topK {
		cumulative += item.val
		if r < cumulative {
			return item.idx
		}
	}
	return topK[len(topK)-1].idx
}

func newTransformerLayer() TransformerLayer {
	return TransformerLayer{
		Wff1: randMatrix(embedDim, embedDim), GradWff1: zeroMatrix(embedDim, embedDim),
		Bff1: make([]float64, embedDim), GradBff1: make([]float64, embedDim),
		Wff2: randMatrix(embedDim, embedDim), GradWff2: zeroMatrix(embedDim, embedDim),
		Bff2: make([]float64, embedDim), GradBff2: make([]float64, embedDim),
		Wq: randMatrix(embedDim, headDim), GradWq: zeroMatrix(embedDim, headDim),
		Wk: randMatrix(embedDim, headDim), GradWk: zeroMatrix(embedDim, headDim),
		Wv: randMatrix(embedDim, headDim), GradWv: zeroMatrix(embedDim, headDim),
		Wout: randMatrix(embedDim, embedDim), GradWout: zeroMatrix(embedDim, embedDim),
	}
}
