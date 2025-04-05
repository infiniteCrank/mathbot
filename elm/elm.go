package elm

import (
	"database/sql"
	"errors"
	"fmt"
	"math"
	"math/rand"
)

// ELM represents a simple Extreme Learning Machine.
type ELM struct {
	InputSize      int
	HiddenSize     int
	OutputSize     int
	InputWeights   [][]float64 // shape: InputSize x HiddenSize
	HiddenBiases   []float64   // length: HiddenSize
	OutputWeights  [][]float64 // shape: HiddenSize x OutputSize
	Activation     int         // 0: Sigmoid, 1: LeakyReLU, 2: Identity
	Regularization float64     // Ridge regularization parameter (lambda)
	ModelType      string
	RMSE           float64
}

// Activation functions.
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func leakyReLU(x float64) float64 {
	if x < 0 {
		return 0.01 * x
	}
	return x
}

// NewELM creates a new ELM with randomly initialized hidden layer parameters.
func NewELM(inputSize, hiddenSize, outputSize, activation int, regularization float64) *ELM {
	elm := &ELM{
		InputSize:      inputSize,
		HiddenSize:     hiddenSize,
		OutputSize:     outputSize,
		Activation:     activation,
		Regularization: regularization,
	}

	// Initialize random input weights.
	elm.InputWeights = make([][]float64, inputSize)
	for i := 0; i < inputSize; i++ {
		elm.InputWeights[i] = make([]float64, hiddenSize)
		for j := 0; j < hiddenSize; j++ {
			elm.InputWeights[i][j] = rand.NormFloat64() // you can scale as needed
		}
	}

	// Initialize random biases for the hidden layer.
	elm.HiddenBiases = make([]float64, hiddenSize)
	for j := 0; j < hiddenSize; j++ {
		elm.HiddenBiases[j] = rand.Float64()
	}

	// Initialize output weights (will be computed in training).
	elm.OutputWeights = make([][]float64, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		elm.OutputWeights[i] = make([]float64, outputSize)
	}

	return elm
}

// hiddenLayer computes the activation output for a single input sample.
func (elm *ELM) hiddenLayer(input []float64) []float64 {
	H := make([]float64, elm.HiddenSize)
	for j := 0; j < elm.HiddenSize; j++ {
		sum := 0.0
		for i := 0; i < elm.InputSize; i++ {
			sum += input[i] * elm.InputWeights[i][j]
		}
		sum += elm.HiddenBiases[j]
		switch elm.Activation {
		case 0: // Sigmoid
			H[j] = sigmoid(sum)
		case 1: // LeakyReLU
			H[j] = leakyReLU(sum)
		case 2: // Identity
			H[j] = sum
		default:
			H[j] = sigmoid(sum)
		}
	}
	return H
}

// Train computes the output weights using ridge regression.
// It assumes trainInputs is an nSamples x InputSize matrix and
// trainTargets is an nSamples x OutputSize matrix.
func (elm *ELM) Train(trainInputs [][]float64, trainTargets [][]float64) {
	nSamples := len(trainInputs)

	// Compute hidden layer output matrix H (nSamples x HiddenSize)
	H := make([][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		H[i] = elm.hiddenLayer(trainInputs[i])
	}

	// Compute H^T * H (HiddenSize x HiddenSize)
	HtH := make([][]float64, elm.HiddenSize)
	for i := 0; i < elm.HiddenSize; i++ {
		HtH[i] = make([]float64, elm.HiddenSize)
		for j := 0; j < elm.HiddenSize; j++ {
			sum := 0.0
			for k := 0; k < nSamples; k++ {
				sum += H[k][i] * H[k][j]
			}
			// Add regularization term on the diagonal.
			if i == j {
				sum += elm.Regularization
			}
			HtH[i][j] = sum
		}
	}

	// Compute H^T * Y (HiddenSize x OutputSize)
	HtY := make([][]float64, elm.HiddenSize)
	for i := 0; i < elm.HiddenSize; i++ {
		HtY[i] = make([]float64, elm.OutputSize)
		for j := 0; j < elm.OutputSize; j++ {
			sum := 0.0
			for k := 0; k < nSamples; k++ {
				sum += H[k][i] * trainTargets[k][j]
			}
			HtY[i][j] = sum
		}
	}

	// Solve for OutputWeights: Beta = (H^T*H)^(-1) * (H^T*Y)
	inv, err := MatrixInverse(HtH)
	if err != nil {
		panic(fmt.Sprintf("Matrix inversion failed: %v", err))
	}
	Beta := MatrixMultiply(inv, HtY)
	elm.OutputWeights = Beta
}

// Predict returns the prediction for a single input sample.
func (elm *ELM) Predict(input []float64) []float64 {
	H := elm.hiddenLayer(input)
	output := make([]float64, elm.OutputSize)
	for j := 0; j < elm.OutputSize; j++ {
		sum := 0.0
		for i := 0; i < elm.HiddenSize; i++ {
			sum += H[i] * elm.OutputWeights[i][j]
		}
		output[j] = sum
	}
	return output
}

// MatrixInverse inverts a square matrix using Gauss-Jordan elimination.
func MatrixInverse(matrix [][]float64) ([][]float64, error) {
	n := len(matrix)
	// Create augmented matrix.
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, 2*n)
		for j := 0; j < n; j++ {
			aug[i][j] = matrix[i][j]
		}
		for j := n; j < 2*n; j++ {
			if j-n == i {
				aug[i][j] = 1
			} else {
				aug[i][j] = 0
			}
		}
	}

	// Gauss-Jordan elimination.
	for i := 0; i < n; i++ {
		// Find pivot.
		pivot := aug[i][i]
		if math.Abs(pivot) < 1e-12 {
			return nil, errors.New("singular matrix")
		}
		// Scale pivot row.
		for j := 0; j < 2*n; j++ {
			aug[i][j] /= pivot
		}
		// Eliminate pivot column in other rows.
		for k := 0; k < n; k++ {
			if k == i {
				continue
			}
			factor := aug[k][i]
			for j := 0; j < 2*n; j++ {
				aug[k][j] -= factor * aug[i][j]
			}
		}
	}

	// Extract inverse from augmented matrix.
	inv := make([][]float64, n)
	for i := 0; i < n; i++ {
		inv[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			inv[i][j] = aug[i][j+n]
		}
	}
	return inv, nil
}

// MatrixMultiply multiplies matrix A (m x n) with matrix B (n x p).
func MatrixMultiply(A, B [][]float64) [][]float64 {
	m := len(A)
	n := len(A[0])
	p := len(B[0])
	C := make([][]float64, m)
	for i := 0; i < m; i++ {
		C[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += A[i][k] * B[k][j]
			}
			C[i][j] = sum
		}
	}
	return C
}

/////////////////////////////
// Database Saving/Loading //
/////////////////////////////

// SaveModel saves the current ELM model to the Postgres database.
// It assumes that the necessary tables (elm, elm_layers, elm_weights) already exist.
func (elm *ELM) SaveModel(db *sql.DB) error {
	// Insert metadata into the elm table including the model_type column.
	var elmID int
	query := `INSERT INTO nn_schema.elm (input_size, hidden_size, output_size, activation, regularization, model_type, rmse)
	          VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id`
	err := db.QueryRow(query, elm.InputSize, elm.HiddenSize, elm.OutputSize, elm.Activation, elm.Regularization, elm.ModelType, elm.RMSE).Scan(&elmID)
	if err != nil {
		return fmt.Errorf("saving elm metadata failed: %v", err)
	}

	// Helper function: save a layer's weights.
	// It inserts a new record into elm_layers then saves each weight in elm_weights.
	saveLayer := func(layerIndex int, matrix [][]float64) error {
		var layerID int
		ql := `INSERT INTO nn_schema.elm_layers (elm_id, layer_index) VALUES ($1, $2) RETURNING id`
		err := db.QueryRow(ql, elmID, layerIndex).Scan(&layerID)
		if err != nil {
			return fmt.Errorf("saving elm layer %d failed: %v", layerIndex, err)
		}
		// Insert each weight.
		ins := `INSERT INTO nn_schema.elm_weights (layer_id, row_index, col_index, weight) VALUES ($1, $2, $3, $4)`
		for i, row := range matrix {
			for j, w := range row {
				_, err := db.Exec(ins, layerID, i, j, w)
				if err != nil {
					return fmt.Errorf("saving weight at layer %d (%d,%d) failed: %v", layerIndex, i, j, err)
				}
			}
		}
		return nil
	}

	// Save layers:
	// Layer 0: InputWeights (shape: InputSize x HiddenSize)
	// Layer 1: HiddenBiases (stored as a 1xHiddenSize matrix)
	// Layer 2: OutputWeights (shape: HiddenSize x OutputSize)
	if err := saveLayer(0, elm.InputWeights); err != nil {
		return err
	}
	// Convert hidden biases into a single-row matrix.
	biasMatrix := make([][]float64, 1)
	biasMatrix[0] = elm.HiddenBiases
	if err := saveLayer(1, biasMatrix); err != nil {
		return err
	}
	if err := saveLayer(2, elm.OutputWeights); err != nil {
		return err
	}

	return nil
}

// LoadModel loads an ELM model from the database given its id.
func LoadModel(db *sql.DB, elmID int) (*ELM, error) {
	// First, load the metadata including the model_type.
	var inputSize, hiddenSize, outputSize, activation int
	var regularization float64
	var modelType string
	query := `SELECT input_size, hidden_size, output_size, activation, regularization, model_type 
	          FROM nn_schema.elm WHERE id = $1`
	err := db.QueryRow(query, elmID).Scan(&inputSize, &hiddenSize, &outputSize, &activation, &regularization, &modelType)
	if err != nil {
		return nil, fmt.Errorf("loading elm metadata failed: %v", err)
	}

	elmModel := &ELM{
		InputSize:      inputSize,
		HiddenSize:     hiddenSize,
		OutputSize:     outputSize,
		Activation:     activation,
		Regularization: regularization,
		ModelType:      modelType,
	}

	// Allocate matrices.
	elmModel.InputWeights = make([][]float64, inputSize)
	for i := 0; i < inputSize; i++ {
		elmModel.InputWeights[i] = make([]float64, hiddenSize)
	}
	elmModel.HiddenBiases = make([]float64, hiddenSize)
	elmModel.OutputWeights = make([][]float64, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		elmModel.OutputWeights[i] = make([]float64, outputSize)
	}

	// Helper function: load a layer given its layer_index.
	loadLayer := func(layerIndex int) ([][]float64, error) {
		// Get the layer id.
		var layerID int
		queryLayer := `SELECT id FROM nn_schema.elm_layers WHERE elm_id = $1 AND layer_index = $2`
		err := db.QueryRow(queryLayer, elmID, layerIndex).Scan(&layerID)
		if err != nil {
			return nil, fmt.Errorf("loading layer %d id failed: %v", layerIndex, err)
		}
		// Determine dimensions.
		var nrows, ncols int
		switch layerIndex {
		case 0:
			nrows, ncols = inputSize, hiddenSize
		case 1:
			nrows, ncols = 1, hiddenSize
		case 2:
			nrows, ncols = hiddenSize, outputSize
		default:
			return nil, fmt.Errorf("invalid layer index %d", layerIndex)
		}
		// Prepare an empty matrix.
		mat := make([][]float64, nrows)
		for i := range mat {
			mat[i] = make([]float64, ncols)
		}
		// Query all weights for this layer.
		queryWeights := `SELECT row_index, col_index, weight FROM nn_schema.elm_weights WHERE layer_id = $1`
		rows, err := db.Query(queryWeights, layerID)
		if err != nil {
			return nil, fmt.Errorf("loading weights for layer %d failed: %v", layerIndex, err)
		}
		defer rows.Close()
		for rows.Next() {
			var r, c int
			var w float64
			if err := rows.Scan(&r, &c, &w); err != nil {
				return nil, err
			}
			if r < nrows && c < ncols {
				mat[r][c] = w
			}
		}
		return mat, nil
	}

	// Load each layer.
	inW, err := loadLayer(0)
	if err != nil {
		return nil, err
	}
	elmModel.InputWeights = inW

	biasMat, err := loadLayer(1)
	if err != nil {
		return nil, err
	}
	elmModel.HiddenBiases = biasMat[0] // biases stored as one row.

	outW, err := loadLayer(2)
	if err != nil {
		return nil, err
	}
	elmModel.OutputWeights = outW

	return elmModel, nil
}
