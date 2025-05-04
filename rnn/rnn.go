// rnn.go - Multi-layer RNN, GRU, LSTM, Transformer Attention in Go
package rnn

import (
	"database/sql"
	"encoding/json"
	"math"
	"math/rand"

	"github.com/infiniteCrank/mathbot/tfidf"
)

func sigmoid(x float64) float64  { return 1.0 / (1.0 + math.Exp(-x)) }
func dsigmoid(y float64) float64 { return y * (1 - y) }
func tanh(x float64) float64     { return math.Tanh(x) }
func dtanh(y float64) float64    { return 1 - y*y }

type GRUModel struct {
	OutputWeights [][]float64
	OutputBias    []float64
}

// Layer interface for RNN-like cells
type Layer interface {
	Forward(x []float64) ([]float64, []float64)
	Reset()
}

// --- RNNCell (Vanilla RNN) ---
type RNNCell struct {
	InputSize, HiddenSize int
	Wih, Whh              [][]float64
	Bh                    []float64
	State                 []float64
}

func NewRNNCell(inputSize, hiddenSize int) *RNNCell {
	return &RNNCell{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		Wih:        RandMatrix(hiddenSize, inputSize),
		Whh:        RandMatrix(hiddenSize, hiddenSize),
		Bh:         RandVector(hiddenSize),
		State:      make([]float64, hiddenSize),
	}
}

func (r *RNNCell) Forward(x []float64) ([]float64, []float64) {
	newState := make([]float64, r.HiddenSize)
	for i := 0; i < r.HiddenSize; i++ {
		sum := r.Bh[i]
		for j := 0; j < r.InputSize; j++ {
			sum += r.Wih[i][j] * x[j]
		}
		for j := 0; j < r.HiddenSize; j++ {
			sum += r.Whh[i][j] * r.State[j]
		}
		newState[i] = tanh(sum)
	}
	r.State = newState
	return newState, newState
}

func (r *RNNCell) Reset() {
	r.State = make([]float64, r.HiddenSize)
}

// --- GRU Cell ---
type GRUCell struct {
	InputSize, HiddenSize int
	Wz, Wr, Wh            [][]float64
	Uz, Ur, Uh            [][]float64
	Bz, Br, Bh            []float64
	State                 []float64
}

func NewGRUCell(inputSize, hiddenSize int) *GRUCell {
	return &GRUCell{
		InputSize: inputSize, HiddenSize: hiddenSize,
		Wz: RandMatrix(hiddenSize, inputSize), Uz: RandMatrix(hiddenSize, hiddenSize), Bz: RandVector(hiddenSize),
		Wr: RandMatrix(hiddenSize, inputSize), Ur: RandMatrix(hiddenSize, hiddenSize), Br: RandVector(hiddenSize),
		Wh: RandMatrix(hiddenSize, inputSize), Uh: RandMatrix(hiddenSize, hiddenSize), Bh: RandVector(hiddenSize),
		State: make([]float64, hiddenSize),
	}
}

func (g *GRUCell) Forward(x []float64) ([]float64, []float64) {
	h := g.State
	r := make([]float64, g.HiddenSize)
	z := make([]float64, g.HiddenSize)
	hTilde := make([]float64, g.HiddenSize)
	newH := make([]float64, g.HiddenSize)
	for i := 0; i < g.HiddenSize; i++ {
		z[i] = sigmoid(Dot(g.Wz[i], x) + Dot(g.Uz[i], h) + g.Bz[i])
		r[i] = sigmoid(Dot(g.Wr[i], x) + Dot(g.Ur[i], h) + g.Br[i])
		hTilde[i] = tanh(Dot(g.Wh[i], x) + Dot(g.Uh[i], Hadamard(r, h)) + g.Bh[i])
		newH[i] = (1-z[i])*h[i] + z[i]*hTilde[i]
	}
	g.State = newH
	return newH, newH
}

func (g *GRUCell) Reset() {
	g.State = make([]float64, g.HiddenSize)
}

// --- LSTM Cell ---
type LSTMCell struct {
	InputSize, HiddenSize  int
	Wf, Wi, Wo, Wc         [][]float64
	Uf, Ui, Uo, Uc         [][]float64
	Bf, Bi, Bo, Bc         []float64
	CellState, HiddenState []float64
}

func NewLSTMCell(inputSize, hiddenSize int) *LSTMCell {
	return &LSTMCell{
		InputSize: inputSize, HiddenSize: hiddenSize,
		Wf: RandMatrix(hiddenSize, inputSize), Uf: RandMatrix(hiddenSize, hiddenSize), Bf: RandVector(hiddenSize),
		Wi: RandMatrix(hiddenSize, inputSize), Ui: RandMatrix(hiddenSize, hiddenSize), Bi: RandVector(hiddenSize),
		Wo: RandMatrix(hiddenSize, inputSize), Uo: RandMatrix(hiddenSize, hiddenSize), Bo: RandVector(hiddenSize),
		Wc: RandMatrix(hiddenSize, inputSize), Uc: RandMatrix(hiddenSize, hiddenSize), Bc: RandVector(hiddenSize),
		CellState: make([]float64, hiddenSize), HiddenState: make([]float64, hiddenSize),
	}
}

func (l *LSTMCell) Forward(x []float64) ([]float64, []float64) {
	h := l.HiddenState
	c := l.CellState
	f := make([]float64, l.HiddenSize)
	i := make([]float64, l.HiddenSize)
	o := make([]float64, l.HiddenSize)
	cTilde := make([]float64, l.HiddenSize)
	for j := 0; j < l.HiddenSize; j++ {
		f[j] = sigmoid(Dot(l.Wf[j], x) + Dot(l.Uf[j], h) + l.Bf[j])
		i[j] = sigmoid(Dot(l.Wi[j], x) + Dot(l.Ui[j], h) + l.Bi[j])
		o[j] = sigmoid(Dot(l.Wo[j], x) + Dot(l.Uo[j], h) + l.Bo[j])
		cTilde[j] = tanh(Dot(l.Wc[j], x) + Dot(l.Uc[j], h) + l.Bc[j])
		c[j] = f[j]*c[j] + i[j]*cTilde[j]
		h[j] = o[j] * tanh(c[j])
	}
	l.CellState, l.HiddenState = c, h
	return h, h
}

func (l *LSTMCell) Reset() {
	l.CellState = make([]float64, l.HiddenSize)
	l.HiddenState = make([]float64, l.HiddenSize)
}

// --- Transformer-style Attention ---
func Attention(query, keys, values [][]float64) []float64 {
	dim := len(query[0])
	scores := make([]float64, len(keys))
	for i := range keys {
		for j := 0; j < dim; j++ {
			scores[i] += query[0][j] * keys[i][j]
		}
	}
	// Softmax
	sumExp := 0.0
	expScores := make([]float64, len(scores))
	for i, s := range scores {
		expScores[i] = math.Exp(s)
		sumExp += expScores[i]
	}
	for i := range expScores {
		expScores[i] /= sumExp
	}
	// Weighted sum
	context := make([]float64, dim)
	for i := range values {
		for j := 0; j < dim; j++ {
			context[j] += expScores[i] * values[i][j]
		}
	}
	return context
}

// --- Utility functions ---

func Hadamard(a, b []float64) []float64 {
	res := make([]float64, len(a))
	for i := range a {
		res[i] = a[i] * b[i]
	}
	return res
}

func RandMatrix(rows, cols int) [][]float64 {
	m := make([][]float64, rows)
	for i := range m {
		m[i] = RandVector(cols)
	}
	return m
}

func RandVector(size int) []float64 {
	v := make([]float64, size)
	for i := range v {
		v[i] = rand.NormFloat64() * 0.01
	}
	return v
}

func SaveModelToPostgres(db *sql.DB, model *GRUModel) error {
	data, err := json.Marshal(model)
	if err != nil {
		return err
	}
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS gru_models (id SERIAL PRIMARY KEY, name TEXT UNIQUE, data JSONB)`)
	if err != nil {
		return err
	}
	_, err = db.Exec(`INSERT INTO gru_models (name, data) VALUES ($1, $2) ON CONFLICT (name) DO UPDATE SET data = EXCLUDED.data`, "markdown_gru", string(data))
	return err
}

func LoadModelFromPostgres(db *sql.DB, name string) (*GRUModel, error) {
	var data string
	err := db.QueryRow(`SELECT data FROM gru_models WHERE name = $1`, name).Scan(&data)
	if err != nil {
		return nil, err
	}
	var model GRUModel
	err = json.Unmarshal([]byte(data), &model)
	return &model, err
}

func CosineSimilarity(a, b []float64) float64 {
	dot, normA, normB := 0.0, 0.0, 0.0
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func FindClosestWord(vec []float64, model *tfidf.TFIDF) string {
	best := ""
	bestScore := -1.0
	for _, word := range model.Vocabulary {
		cand := model.Transform(word)
		score := CosineSimilarity(vec, cand)
		if score > bestScore {
			bestScore = score
			best = word
		}
	}
	return best
}

func Dot(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
