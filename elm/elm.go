package elm // Define the package name as 'elm'

import (
	"database/sql"  // Importing the sql package for database interactions
	"encoding/json" // Importing the json package for JSON encoding and decoding
	"errors"        // Importing the errors package for handling error messages
	"fmt"           // Importing the fmt package for formatted I/O operations
	"math"          // Importing the math package for mathematical functions
	"math/rand"     // Importing the rand package for generating pseudo-random numbers
	"os"            // Importing the os package for operating system functionalities like file handling
)

// ELM represents a simple Extreme Learning Machine.
type ELM struct {
	InputSize      int         // Number of input features
	HiddenSize     int         // Number of neurons in the hidden layer
	OutputSize     int         // Number of output features
	InputWeights   [][]float64 // Matrix of input weights for the hidden layer (InputSize x HiddenSize)
	HiddenBiases   []float64   // Vector of biases for the hidden layer (length: HiddenSize)
	OutputWeights  [][]float64 // Matrix of weights for outputs (HiddenSize x OutputSize)
	Activation     int         // Activation function type (0: Sigmoid, 1: LeakyReLU, 2: Identity)
	Regularization float64     // Ridge regularization parameter (lambda)
	ModelType      string      // Type of the model being used
	RMSE           float64     // Root Mean Square Error for the current model
}

// Activation functions.
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x)) // Sigmoid activation function
}

func leakyReLU(x float64) float64 {
	if x < 0 {
		return 0.01 * x // Leaky ReLU: If the input is negative, return a small linear value
	}
	return x // If positive, return the input as is
}

// NewELM creates a new ELM with randomly initialized hidden layer parameters.
func NewELM(inputSize, hiddenSize, outputSize, activation int, regularization float64) *ELM {
	elm := &ELM{ // Create a new ELM instance
		InputSize:      inputSize,
		HiddenSize:     hiddenSize,
		OutputSize:     outputSize,
		Activation:     activation,
		Regularization: regularization,
	}

	// Initialize random input weights.
	elm.InputWeights = make([][]float64, inputSize) // Create a slice for input weights
	for i := 0; i < inputSize; i++ {
		elm.InputWeights[i] = make([]float64, hiddenSize) // Create a slice for weights for each input
		for j := 0; j < hiddenSize; j++ {
			elm.InputWeights[i][j] = rand.NormFloat64() // Initialize with random values from a normal distribution
		}
	}

	// Initialize random biases for the hidden layer.
	elm.HiddenBiases = make([]float64, hiddenSize) // Create a slice for hidden biases
	for j := 0; j < hiddenSize; j++ {
		elm.HiddenBiases[j] = rand.Float64() // Set biases to random values between 0 and 1
	}

	// Initialize output weights (will be computed in training).
	elm.OutputWeights = make([][]float64, hiddenSize) // Create a slice for output weights
	for i := 0; i < hiddenSize; i++ {
		elm.OutputWeights[i] = make([]float64, outputSize) // Create a sub-slice for weights for each hidden neuron
	}

	return elm // Return the newly created ELM instance
}

// HiddenLayer computes the activation output for a single input sample.
func (elm *ELM) HiddenLayer(input []float64) []float64 {
	H := make([]float64, elm.HiddenSize) // Create a slice for hidden layer outputs
	for j := 0; j < elm.HiddenSize; j++ {
		sum := 0.0 // Initialize sum for weighted inputs
		for i := 0; i < elm.InputSize; i++ {
			sum += input[i] * elm.InputWeights[i][j] // Compute weighted input sum
		}
		sum += elm.HiddenBiases[j] // Add the bias for the neuron
		switch elm.Activation {    // Apply the activation function based on the specified type
		case 0: // Sigmoid activation
			H[j] = sigmoid(sum)
		case 1: // Leaky ReLU activation
			H[j] = leakyReLU(sum)
		case 2: // Identity activation
			H[j] = sum
		default:
			H[j] = sigmoid(sum) // Default to sigmoid if activation type is unknown
		}
	}
	return H // Return the activation output for the hidden layer
}

// Train computes the output weights using ridge regression.
// It assumes trainInputs is an nSamples x InputSize matrix and
func (elm *ELM) Train(trainInputs [][]float64, trainTargets [][]float64, valInputs [][]float64, valTargets [][]float64) {
	elm.TrainEpoch(trainInputs, trainTargets)
	valLoss := elm.CalculateLoss(valInputs, valTargets)
	fmt.Printf("Validation Loss: %.4f\n", valLoss)
}

// CalculateLoss computes loss on given inputs and targets
func (elm *ELM) CalculateLoss(inputs [][]float64, targets [][]float64) float64 {
	// Calculate mean squared error (MSE) over the given validation set
	var totalLoss float64
	for i := 0; i < len(inputs); i++ {
		pred := elm.Predict(inputs[i])
		for j := 0; j < len(pred); j++ {
			totalLoss += math.Pow(pred[j]-targets[i][j], 2) // MSE calculation
		}
	}
	return totalLoss / float64(len(inputs)) // Return average loss
}

// TrainEpoch performs a single epoch of training using the input-output pairs
func (elm *ELM) TrainEpoch(trainInputs [][]float64, trainTargets [][]float64) {
	nSamples := len(trainInputs)

	// Compute hidden layer output matrix H (nSamples x HiddenSize)
	H := make([][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		H[i] = elm.HiddenLayer(trainInputs[i]) // Get hidden layer output
	}

	// Compute H^T * H (HiddenSize x HiddenSize)
	HtH := make([][]float64, elm.HiddenSize)
	for i := 0; i < elm.HiddenSize; i++ {
		HtH[i] = make([]float64, elm.HiddenSize)
		for j := 0; j < elm.HiddenSize; j++ {
			sum := 0.0
			for k := 0; k < nSamples; k++ {
				sum += H[k][i] * H[k][j] // Compute dot products
			}
			// Add regularization term on the diagonal
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
				sum += H[k][i] * trainTargets[k][j] // Compute dot products for H^T * Y
			}
			HtY[i][j] = sum
		}
	}

	// Solve for OutputWeights: Beta = (H^T*H)^(-1) * (H^T*Y)
	inv, err := MatrixInverse(HtH) // Invert the matrix H^T * H
	if err != nil {
		panic(fmt.Sprintf("Matrix inversion failed: %v", err)) // Handle error for matrix inversion
	}
	elm.OutputWeights = MatrixMultiply(inv, HtY) // Multiply the inverted matrix with H^T * Y to get output weights
}

// Predict returns the prediction for a single input sample.
func (elm *ELM) Predict(input []float64) []float64 {
	H := elm.HiddenLayer(input)               // Compute the hidden layer activations for the given input
	output := make([]float64, elm.OutputSize) // Create a slice for the output predictions
	for j := 0; j < elm.OutputSize; j++ {
		sum := 0.0 // Initialize sum for the output computation
		for i := 0; i < elm.HiddenSize; i++ {
			sum += H[i] * elm.OutputWeights[i][j] // Compute the dot product for the output
		}
		output[j] = sum // Assign the computed value to the output
	}
	return output // Return the final predictions
}

// MatrixInverse inverts a square matrix using Gauss-Jordan elimination.
func MatrixInverse(matrix [][]float64) ([][]float64, error) {
	n := len(matrix) // Get the size of the matrix
	// Create augmented matrix.
	aug := make([][]float64, n) // Create an augmented matrix with 2n columns
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, 2*n) // Initialize each row of the augmented matrix
		for j := 0; j < n; j++ {      // Copy the original matrix on the left side
			aug[i][j] = matrix[i][j]
		}
		for j := n; j < 2*n; j++ { // Initialize the right side of the augmented matrix as the identity matrix
			if j-n == i {
				aug[i][j] = 1 // Set the diagonal elements to 1
			} else {
				aug[i][j] = 0 // Set all other elements to 0
			}
		}
	}

	// Gauss-Jordan elimination.
	for i := 0; i < n; i++ {
		// Find pivot.
		pivot := aug[i][i]           // Find the pivot element for the current row
		if math.Abs(pivot) < 1e-12 { // Check for singular matrix
			return nil, errors.New("singular matrix") // Return an error if the matrix is singular
		}
		// Scale the pivot row.
		for j := 0; j < 2*n; j++ {
			aug[i][j] /= pivot // Normalize the pivot row
		}
		// Eliminate pivot column in other rows.
		for k := 0; k < n; k++ {
			if k == i { // Skip the pivot row
				continue
			}
			factor := aug[k][i] // Get the factor for elimination
			for j := 0; j < 2*n; j++ {
				aug[k][j] -= factor * aug[i][j] // Eliminate the current column in other rows
			}
		}
	}

	// Extract inverse from augmented matrix.
	inv := make([][]float64, n) // Create a matrix for the inverse
	for i := 0; i < n; i++ {
		inv[i] = make([]float64, n) // Initialize each row of the inverse matrix
		for j := 0; j < n; j++ {
			inv[i][j] = aug[i][j+n] // Get the inverse from the right side of the augmented matrix
		}
	}
	return inv, nil // Return the computed inverse matrix
}

// MatrixMultiply multiplies matrix A (m x n) with matrix B (n x p).
func MatrixMultiply(A, B [][]float64) [][]float64 {
	m := len(A)               // Number of rows in matrix A
	n := len(A[0])            // Number of columns in matrix A
	p := len(B[0])            // Number of columns in matrix B
	C := make([][]float64, m) // Create a result matrix C with dimensions m x p
	for i := 0; i < m; i++ {
		C[i] = make([]float64, p) // Initialize each row of the result matrix
		for j := 0; j < p; j++ {
			sum := 0.0 // Initialize sum for the current element
			for k := 0; k < n; k++ {
				sum += A[i][k] * B[k][j] // Compute the dot product for the current element
			}
			C[i][j] = sum // Assign the computed value to the result matrix
		}
	}
	return C // Return the resulting matrix
}

/////////////////////////////
// Database Saving/Loading //
/////////////////////////////

// SaveModel saves the current ELM model to the Postgres database.
// It assumes that the necessary tables (elm, elm_layers, elm_weights) already exist.
func (elm *ELM) SaveModel(db *sql.DB) error {
	// Insert metadata into the elm table including the model_type column.
	var elmID int // Variable to store the generated ID of the ELM record
	query := `INSERT INTO nn_schema.elm (input_size, hidden_size, output_size, activation, regularization, model_type, rmse)
	          VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id` // SQL query to insert the model metadata
	err := db.QueryRow(query, elm.InputSize, elm.HiddenSize, elm.OutputSize, elm.Activation, elm.Regularization, elm.ModelType, elm.RMSE).Scan(&elmID) // Execute the query and get the generated ID
	if err != nil {
		return fmt.Errorf("saving elm metadata failed: %v", err) // Return error if insertion fails
	}

	// Helper function: save a layer's weights.
	// It inserts a new record into elm_layers then saves each weight in elm_weights.
	saveLayer := func(layerIndex int, matrix [][]float64) error {
		var layerID int                                                                             // Variable to store the generated ID of the layer record
		ql := `INSERT INTO nn_schema.elm_layers (elm_id, layer_index) VALUES ($1, $2) RETURNING id` // SQL query to insert layer metadata
		err := db.QueryRow(ql, elmID, layerIndex).Scan(&layerID)                                    // Execute the query and get the generated layer ID
		if err != nil {
			return fmt.Errorf("saving elm layer %d failed: %v", layerIndex, err) // Return error if insertion fails
		}
		// Insert each weight.
		ins := `INSERT INTO nn_schema.elm_weights (layer_id, row_index, col_index, weight) VALUES ($1, $2, $3, $4)` // SQL query to insert weights
		for i, row := range matrix {                                                                                // Iterate over each row of the weight matrix
			for j, w := range row { // Iterate over each weight in the row
				_, err := db.Exec(ins, layerID, i, j, w) // Execute the weight insertion query
				if err != nil {
					return fmt.Errorf("saving weight at layer %d (%d,%d) failed: %v", layerIndex, i, j, err) // Return error if insertion fails
				}
			}
		}
		return nil // Return nil if saving the layer was successful
	}

	// Save layers:
	// Layer 0: InputWeights (shape: InputSize x HiddenSize)
	// Layer 1: HiddenBiases (stored as a 1xHiddenSize matrix)
	// Layer 2: OutputWeights (shape: HiddenSize x OutputSize)
	if err := saveLayer(0, elm.InputWeights); err != nil {
		return err // Save input weights and return on error
	}
	// Convert hidden biases into a single-row matrix.
	biasMatrix := make([][]float64, 1) // Create a single-row matrix for biases
	biasMatrix[0] = elm.HiddenBiases   // Assign hidden biases to the matrix
	if err := saveLayer(1, biasMatrix); err != nil {
		return err // Save hidden biases and return on error
	}
	if err := saveLayer(2, elm.OutputWeights); err != nil {
		return err // Save output weights and return on error
	}

	return nil // Return nil if all layers were saved successfully
}

// LoadModel loads an ELM model from the database given its id.
func LoadModel(db *sql.DB, elmID int) (*ELM, error) {
	// First, load the metadata including the model_type.
	var inputSize, hiddenSize, outputSize, activation int // Variables to hold model metadata
	var regularization float64
	var modelType string
	query := `SELECT input_size, hidden_size, output_size, activation, regularization, model_type 
	          FROM nn_schema.elm WHERE id = $1` // SQL query to fetch model metadata
	err := db.QueryRow(query, elmID).Scan(&inputSize, &hiddenSize, &outputSize, &activation, &regularization, &modelType) // Execute query and scan results into variables
	if err != nil {
		return nil, fmt.Errorf("loading elm metadata failed: %v", err) // Return error if loading metadata fails
	}

	// Create a new ELM instance with the loaded metadata.
	elmModel := &ELM{
		InputSize:      inputSize,
		HiddenSize:     hiddenSize,
		OutputSize:     outputSize,
		Activation:     activation,
		Regularization: regularization,
		ModelType:      modelType,
	}

	// Allocate matrices for weights.
	elmModel.InputWeights = make([][]float64, inputSize) // Allocate memory for input weights
	for i := 0; i < inputSize; i++ {
		elmModel.InputWeights[i] = make([]float64, hiddenSize) // Initialize each row
	}
	elmModel.HiddenBiases = make([]float64, hiddenSize)    // Allocate memory for hidden biases
	elmModel.OutputWeights = make([][]float64, hiddenSize) // Allocate memory for output weights
	for i := 0; i < hiddenSize; i++ {
		elmModel.OutputWeights[i] = make([]float64, outputSize) // Initialize each row for output weights
	}

	// Helper function: load a layer given its layer_index.
	loadLayer := func(layerIndex int) ([][]float64, error) {
		// Get the layer id.
		var layerID int                                                                            // Variable to store the layer ID
		queryLayer := `SELECT id FROM nn_schema.elm_layers WHERE elm_id = $1 AND layer_index = $2` // SQL query to fetch layer ID
		err := db.QueryRow(queryLayer, elmID, layerIndex).Scan(&layerID)                           // Execute query and scan
		if err != nil {
			return nil, fmt.Errorf("loading layer %d id failed: %v", layerIndex, err) // Return error if fetching layer ID fails
		}
		// Determine dimensions.
		var nrows, ncols int
		switch layerIndex { // Determine the dimensions based on the layer index
		case 0: // Input weights layer
			nrows, ncols = inputSize, hiddenSize
		case 1: // Hidden biases layer
			nrows, ncols = 1, hiddenSize
		case 2: // Output weights layer
			nrows, ncols = hiddenSize, outputSize
		default:
			return nil, fmt.Errorf("invalid layer index %d", layerIndex) // Return error if layer index is invalid
		}
		// Prepare an empty matrix.
		mat := make([][]float64, nrows) // Create an empty matrix
		for i := range mat {
			mat[i] = make([]float64, ncols) // Initialize each row with appropriate column size
		}
		// Query all weights for this layer.
		queryWeights := `SELECT row_index, col_index, weight FROM nn_schema.elm_weights WHERE layer_id = $1` // SQL query to fetch weights for the layer
		rows, err := db.Query(queryWeights, layerID)                                                         // Execute the weight query
		if err != nil {
			return nil, fmt.Errorf("loading weights for layer %d failed: %v", layerIndex, err) // Return error if fetching weights fails
		}
		defer rows.Close() // Ensure rows are closed after processing
		for rows.Next() {  // Iterate over the fetched weights
			var r, c int
			var w float64                                 // Variables to hold row index, column index, and weight value
			if err := rows.Scan(&r, &c, &w); err != nil { // Scan the row into the variables
				return nil, err // Return error if scanning fails
			}
			if r < nrows && c < ncols { // Check bounds
				mat[r][c] = w // Assign the weight to the appropriate position in the matrix
			}
		}
		return mat, nil // Return the loaded layer weights
	}

	// Load each layer.
	inW, err := loadLayer(0) // Load input weights
	if err != nil {
		return nil, err // Return error if loading input weights fails
	}
	elmModel.InputWeights = inW // Assign loaded input weights to the model

	biasMat, err := loadLayer(1) // Load hidden biases
	if err != nil {
		return nil, err // Return error if loading hidden biases fails
	}
	elmModel.HiddenBiases = biasMat[0] // Assign hidden biases (stored as a single row) to the model

	outW, err := loadLayer(2) // Load output weights
	if err != nil {
		return nil, err // Return error if loading output weights fails
	}
	elmModel.OutputWeights = outW // Assign loaded output weights to the model

	return elmModel, nil // Return the fully loaded ELM model
}

// ExportToFile exports the model to a JSON file.
func (elm *ELM) ExportToFile(filename string) error {
	// Convert the ELM model structure to JSON (or your preferred format)
	data, err := json.Marshal(elm) // Marshal the ELM model to JSON format
	if err != nil {
		return fmt.Errorf("error marshaling model to JSON: %v", err) // Return error if marshaling fails
	}
	return os.WriteFile(filename, data, 0644) // Write the JSON data to a file with write permissions
}

// ImportModelFromFile imports a model from a JSON file.
func ImportModelFromFile(filename string) (*ELM, error) {
	data, err := os.ReadFile(filename) // Read the file contents into data
	if err != nil {
		return nil, fmt.Errorf("error reading file: %v", err) // Return error if file reading fails
	}

	elm := &ELM{}                                     // Create a new ELM model instance
	if err := json.Unmarshal(data, elm); err != nil { // Unmarshal the JSON data into the model
		return nil, fmt.Errorf("error unmarshaling JSON to model: %v", err) // Return error if unmarshalling fails
	}
	return elm, nil // Return the imported ELM model
}

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
