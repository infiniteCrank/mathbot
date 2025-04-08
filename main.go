package main

import (
	"bufio"
	"database/sql"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/infiniteCrank/mathbot/NeuralNetwork"
	"github.com/infiniteCrank/mathbot/db"
	"github.com/infiniteCrank/mathbot/elm"
	_ "github.com/lib/pq"
)

//////////////////////////
// Database Table Setup //
//////////////////////////

// createTables creates the necessary tables in Postgres if they don't already exist.
func createTables(db *sql.DB) error {
	tx, err := db.Begin()
	if err != nil {
		return fmt.Errorf("begin transaction failed: %v", err)
	}
	queries := []string{
		`CREATE TABLE IF NOT EXISTS nn_schema.elm (
			id SERIAL PRIMARY KEY,
			input_size INT,
			hidden_size INT,
			output_size INT,
			activation INT,
			regularization FLOAT8,
			rmse FLOAT8,
			model_type TEXT
		);`,
		`CREATE TABLE IF NOT EXISTS nn_schema.elm_layers (
			id SERIAL PRIMARY KEY,
			elm_id INT REFERENCES nn_schema.elm(id) ON DELETE CASCADE,
			layer_index INT
		);`,
		`CREATE TABLE IF NOT EXISTS nn_schema.elm_weights (
			id SERIAL PRIMARY KEY,
			layer_id INT REFERENCES nn_schema.elm_layers(id) ON DELETE CASCADE,
			row_index INT,
			col_index INT,
			weight FLOAT8
		);`,
	}
	for _, q := range queries {
		if _, err := tx.Exec(q); err != nil {
			tx.Rollback()
			return fmt.Errorf("creating table failed: %v", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit transaction failed: %v", err)
	}
	return nil
}

// dropTables drops all the elm tables.
func dropTables(db *sql.DB) error {
	queries := []string{
		`DROP TABLE IF EXISTS nn_schema.elm_weights CASCADE;`,
		`DROP TABLE IF EXISTS nn_schema.elm_layers CASCADE;`,
		`DROP TABLE IF EXISTS nn_schema.elm CASCADE;`,
	}
	for _, q := range queries {
		if _, err := db.Exec(q); err != nil {
			return fmt.Errorf("dropping table failed: %v", err)
		}
	}
	return nil
}

/////////////////////////
// Utility Functions   //
/////////////////////////

// parseInput parses a comma-separated list of numbers from a string.
func parseInput(inputStr string) ([]float64, error) {
	parts := strings.Split(inputStr, ",")
	var nums []float64
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		val, err := strconv.ParseFloat(p, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid number '%s': %v", p, err)
		}
		nums = append(nums, val)
	}
	return nums, nil
}

// listModels queries and lists all saved models and their types.
func listModels(dbConn *sql.DB) error {
	rows, err := dbConn.Query(`SELECT id, input_size, hidden_size, output_size, activation, regularization, model_type, rmse FROM nn_schema.elm`)
	if err != nil {
		return fmt.Errorf("error querying models: %v", err)
	}
	defer rows.Close()

	fmt.Println("Saved Models:")
	for rows.Next() {
		var id, inputSize, hiddenSize, outputSize, activation int
		var regularization float64
		var rmse float64
		var modelType string
		if err := rows.Scan(&id, &inputSize, &hiddenSize, &outputSize, &activation, &regularization, &modelType, &rmse); err != nil {
			return fmt.Errorf("error scanning model row: %v", err)
		}
		fmt.Printf("ID: %d | Type: %s | Input Size: %d | Hidden Size: %d | Output Size: %d | Activation: %d | Regularization: %.4f | RMSE: %.4f\n",
			id, modelType, inputSize, hiddenSize, outputSize, activation, regularization, rmse)
	}
	return nil
}

// generateDataset is the existing helper for simple math functions.
func generateDataset(funcType string, samples int) ([][]float64, [][]float64) {
	inputs := make([][]float64, samples)
	outputs := make([][]float64, samples)

	for i := 0; i < samples; i++ {
		x := rand.Float64() * 2 * math.Pi // Range: [0, 2π]
		inputs[i] = []float64{x}

		var y float64
		if funcType == "sin" {
			y = math.Sin(x)
		} else if funcType == "exp" {
			y = math.Exp(x)
		} else {
			y = 0
		}
		outputs[i] = []float64{y}
	}

	return inputs, outputs
}

func generateMixedCountingData(steps []int, samplesPerStep int) ([][]float64, []float64) {
	inputs := [][]float64{}
	targets := []float64{}

	for _, step := range steps {
		start := 0
		for j := 0; j < samplesPerStep; j++ {
			base := start + j*step
			seq := []float64{
				float64(base),
				float64(base + step),
				float64(base + 2*step),
				float64(base + 3*step),
				float64(base + 4*step),
			}
			target := float64(base + 5*step)
			inputs = append(inputs, seq)
			targets = append(targets, target)
		}
	}
	return inputs, targets
}

func reshapeTargetsFlatTo2D(flat []float64, outputDim int) ([][]float64, error) {
	if len(flat)%outputDim != 0 {
		return nil, fmt.Errorf("cannot reshape: %d values into output dimension %d", len(flat), outputDim)
	}
	samples := len(flat) / outputDim
	result := make([][]float64, samples)
	for i := 0; i < samples; i++ {
		start := i * outputDim
		result[i] = flat[start : start+outputDim]
	}
	return result, nil
}

//////////////////////////
// Protein Data Section //
//////////////////////////

// ProteinSample represents a simple protein with its amino acid sequence and corresponding secondary structure.
type ProteinSample struct {
	Sequence  string
	Structure string
}

// aminoAcidMap maps standard amino acids to an index (0 to 19).
var aminoAcidMap = map[rune]int{
	'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
	'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
	'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
	'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
}

// encodeAminoAcid returns a one-hot vector (length 20) for a given amino acid.
func encodeAminoAcid(aa rune) []float64 {
	vec := make([]float64, 20)
	if idx, ok := aminoAcidMap[aa]; ok {
		vec[idx] = 1.0
	}
	return vec
}

// encodeWindow encodes a sliding window (of size windowSize) around position pos in sequence.
// It returns a flattened vector of length windowSize*20. Positions outside the sequence are padded with zeros.
func encodeWindow(sequence string, pos, windowSize int) []float64 {
	half := windowSize / 2
	encoded := make([]float64, 0, windowSize*20)
	seqRunes := []rune(sequence)
	seqLen := len(seqRunes)
	for i := pos - half; i <= pos+half; i++ {
		if i < 0 || i >= seqLen {
			// Padding with zero vector.
			encoded = append(encoded, make([]float64, 20)...)
		} else {
			encoded = append(encoded, encodeAminoAcid(seqRunes[i])...)
		}
	}
	return encoded
}

// structureMap defines one-hot encoding for secondary structure labels.
var structureMap = map[rune][]float64{
	'H': {1, 0, 0}, // Helix
	'E': {0, 1, 0}, // Sheet
	'C': {0, 0, 1}, // Coil (or other)
}

// generateProteinDataset processes a small hardcoded dataset into training examples.
// It uses a sliding window approach to create an input vector from the amino acid sequence
// and the corresponding one-hot label from the structure string for the center residue.
func generateProteinDataset(windowSize int) ([][]float64, [][]float64) {
	// Two small hardcoded protein samples with matching sequence and structure lengths.
	dataset := []ProteinSample{
		{
			Sequence:  "ACDEFGHIKLMNPQRSTVWY", // 20 residues
			Structure: "HHHHEEEECCCCCHHHHEEE", // 20 characters
		},
		{
			Sequence:  "MKTIIALSYIFCLVFADYKDDDDK", // 24 residues
			Structure: "CCCCCHHHHCCCCEEEECCCCCCC", // 24 characters (5+4+4+4+7=24)
		},
	}

	var inputs [][]float64
	var outputs [][]float64

	// For each sample, generate examples using a sliding window.
	for _, sample := range dataset {
		seqRunes := []rune(sample.Sequence)
		structRunes := []rune(sample.Structure)
		seqLen := len(seqRunes)
		if seqLen != len(structRunes) {
			fmt.Printf("Warning: sequence and structure lengths do not match for sample %v\n", sample)
			continue
		}
		for pos := 0; pos < seqLen; pos++ {
			// Create input vector for this position.
			inputVec := encodeWindow(sample.Sequence, pos, windowSize)
			// Get the one-hot label for the center residue.
			centerStructure := structRunes[pos]
			label, ok := structureMap[centerStructure]
			if !ok {
				// If unknown structure letter, default to coil.
				label = structureMap['C']
			}
			inputs = append(inputs, inputVec)
			outputs = append(outputs, label)
		}
	}
	return inputs, outputs
}

//////////////////////////
// Main Function        //
//////////////////////////

func main() {
	rand.Seed(time.Now().UnixNano())

	// Mode flag supports: addnew, addpredict, countnew, countpredict, list, drop, combineTech, protein.
	mode := flag.String("mode", "addnew", "Mode: 'addnew', 'addpredict', 'countnew', 'countpredict', 'list', 'drop', 'combineTech', or 'protein'")
	// For loading/predicting, provide the model ID.
	modelID := flag.Int("id", 0, "ID of the model to load (used with addpredict, countpredict, or list)")
	// For prediction modes, provide input numbers.
	inputStr := flag.String("input", "", "Comma-separated list of numbers as input")
	flag.Parse()

	dbConn := db.ConnectDB()
	defer dbConn.Close()

	// If mode is "drop", ask for confirmation then drop all tables.
	if *mode == "drop" {
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("Are you sure you want to drop all elm tables? This action cannot be undone. (y/n): ")
		answer, _ := reader.ReadString('\n')
		answer = strings.TrimSpace(strings.ToLower(answer))
		if answer == "y" || answer == "yes" {
			if err := dropTables(dbConn); err != nil {
				fmt.Printf("Error dropping tables: %v\n", err)
				os.Exit(1)
			}
			fmt.Println("All elm tables dropped successfully.")
		} else {
			fmt.Println("Aborted table drop.")
		}
		os.Exit(0)
	}

	// Create tables if they do not exist.
	if err := createTables(dbConn); err != nil {
		fmt.Printf("Error creating tables: %v\n", err)
		os.Exit(1)
	}

	switch *mode {
	case "combineTech":
		// Existing combineTech mode using ELM+Neural Network for sin(x)
		trainInputs, trainOutputs := generateDataset("sin", 100000)
		elmModel := elm.NewELM(1, 20, 1, 0, 0.01) // 1 input, 20 hidden neurons, 1 output
		elmModel.Train(trainInputs, trainOutputs)
		transformedFeatures := make([][]float64, len(trainInputs))
		for i, input := range trainInputs {
			transformedFeatures[i] = elmModel.HiddenLayer(input)
		}
		layerSizes := []int{20, 20, 1} // 20 features from ELM, 20 hidden neurons, 1 output
		activations := []int{NeuralNetwork.LeakyReLUActivation, NeuralNetwork.IdentityActivation}
		nnModel := NeuralNetwork.NewNeuralNetwork(layerSizes, activations, 0.00001, 0.01)
		nnModel.Train(transformedFeatures, trainOutputs, 1000, 0.95, 100, 32)
		testInputs, _ := generateDataset("sin", 100)
		for _, input := range testInputs {
			transformed := elmModel.HiddenLayer(input)
			predicted := nnModel.PredictRegression(transformed)
			fmt.Printf("Input: %.2f, Predicted: %.4f\n", input[0], predicted[0])
		}

	case "protein":
		// Protein structure prediction using ELM.
		windowSize := 15 // Use an odd number for a centered window.
		inputs, outputs := generateProteinDataset(windowSize)
		fmt.Printf("Generated %d protein training examples.\n", len(inputs))
		// Input size is windowSize * 20 (one-hot per amino acid)
		inputSize := windowSize * 20
		hiddenSize := 50 // adjust as needed
		outputSize := 3  // three secondary structure classes: H, E, C
		activation := 0  // 0 for sigmoid activation (from your elm code)
		regularization := 0.01
		proteinModel := elm.NewELM(inputSize, hiddenSize, outputSize, activation, regularization)
		proteinModel.ModelType = "protein_structure"
		fmt.Println("Training protein structure prediction model using ELM...")
		proteinModel.Train(inputs, outputs)
		// Evaluate on the training set.
		correct := 0
		for i, input := range inputs {
			pred := proteinModel.Predict(input)
			// Get predicted class as index of the max value.
			if argMax(pred) == argMax(outputs[i]) {
				correct++
			}
		}
		accuracy := float64(correct) / float64(len(inputs)) * 100.0
		fmt.Printf("Training accuracy: %.2f%%\n", accuracy)
		// Optionally, save the model if accuracy is acceptable.
		if accuracy > 70.0 { // adjust threshold as needed
			fmt.Println("Accuracy acceptable. Saving protein model to database...")
			if err := proteinModel.SaveModel(dbConn); err != nil {
				fmt.Printf("Error saving protein model: %v\n", err)
			} else {
				fmt.Println("Protein model saved successfully.")
			}
		} else {
			fmt.Println("Accuracy too low. Model not saved.")
		}

	case "list":
		listModels(dbConn)

	// ------------------ Addition Modes ------------------
	case "addnew":
		numSamples := 1000
		var trainingInputs [][]float64
		var trainingTargets [][]float64
		inputMax := 100.0
		outputMax := 200.0
		for i := 0; i < numSamples; i++ {
			a := rand.Float64() * inputMax
			b := rand.Float64() * inputMax
			trainingInputs = append(trainingInputs, []float64{a / inputMax, b / inputMax})
			trainingTargets = append(trainingTargets, []float64{(a + b) / outputMax})
		}
		fmt.Printf("Generated %d training examples for addition.\n", numSamples)
		addModel := elm.NewELM(2, 10, 1, 0, 0.001)
		addModel.ModelType = "addition"
		fmt.Println("Training new addition model...")
		addModel.Train(trainingInputs, trainingTargets)
		fmt.Println("Training completed.")
		var mseSum float64
		for i := 0; i < 100; i++ {
			a := rand.Float64() * inputMax
			b := rand.Float64() * inputMax
			testInput := []float64{a / inputMax, b / inputMax}
			pred := addModel.Predict(testInput)
			predictedSum := pred[0] * outputMax
			trueSum := a + b
			diff := predictedSum - trueSum
			mseSum += diff * diff
		}
		rmse := math.Sqrt(mseSum / 100)
		addModel.RMSE = rmse
		fmt.Printf("Evaluation RMSE for addition: %.6f\n", rmse)
		rmseThreshold := 5.0
		if rmse < rmseThreshold {
			fmt.Println("RMSE acceptable. Saving addition model to database...")
			if err := addModel.SaveModel(dbConn); err != nil {
				fmt.Printf("Error saving model: %v\n", err)
			} else {
				fmt.Println("Addition model saved successfully.")
			}
		} else {
			fmt.Println("RMSE too high. Model not saved.")
		}

	case "addpredict":
		if *modelID <= 0 {
			fmt.Println("Please provide a valid model ID using -id flag for addition prediction.")
			os.Exit(1)
		}
		if *inputStr == "" {
			fmt.Println("Please provide a comma-separated list of 2 numbers using -input flag.")
			os.Exit(1)
		}
		nums, err := parseInput(*inputStr)
		if err != nil {
			fmt.Printf("Error parsing input: %v\n", err)
			os.Exit(1)
		}
		if len(nums) != 2 {
			fmt.Println("Please provide exactly 2 numbers for addition prediction.")
			os.Exit(1)
		}
		loadedModel, err := elm.LoadModel(dbConn, *modelID)
		if err != nil {
			fmt.Printf("Error loading addition model: %v\n", err)
			os.Exit(1)
		}
		normalizedInput := []float64{nums[0] / 100.0, nums[1] / 100.0}
		pred := loadedModel.Predict(normalizedInput)
		predictedSum := pred[0] * 200.0
		fmt.Printf("For input %.2f + %.2f, the predicted sum is: %.4f\n", nums[0], nums[1], predictedSum)

	// ------------------ Counting Modes ------------------
	case "countnew":
		startNum := 1
		endNum := 6000
		var trainingInputs [][]float64
		var trainingTargets [][]float64
		for i := startNum; i <= endNum-5; i++ {
			sequence := []float64{
				float64(i) + rand.Float64()*0.1,
				float64(i+1) + rand.Float64()*0.1,
				float64(i+2) + rand.Float64()*0.1,
				float64(i+3) + rand.Float64()*0.1,
				float64(i+4) + rand.Float64()*0.1,
			}
			trainingInputs = append(trainingInputs, sequence)
			trainingTargets = append(trainingTargets, []float64{float64(i + 5)})
		}
		maxValue := float64(endNum + 5)
		// Normalize sequential counting data.
		for i := range trainingInputs {
			for j := range trainingInputs[i] {
				trainingInputs[i][j] /= maxValue
			}
		}
		for i := range trainingTargets {
			trainingTargets[i][0] /= maxValue
		}
		fmt.Printf("Generated %d training examples for counting.\n", len(trainingInputs))

		// Generate mixed counting data.
		steps := []int{2, 8, 12, 16, 32, 64, 10}
		mixedInputs, mixedOutput := generateMixedCountingData(steps, 100)
		fmt.Printf("Generated %d training examples for counting (mixed data).\n", len(mixedInputs))

		countModel := elm.NewELM(5, 50, 1, 0, 0.001)
		countModel.ModelType = "counting"
		fmt.Println("Training new counting model on sequential data...")
		countModel.Train(trainingInputs, trainingTargets)
		fmt.Println("Training completed on sequential data.")

		// Process the mixed data: reshape and normalize.
		reshapedOutput, err := reshapeTargetsFlatTo2D(mixedOutput, 1)
		if err != nil {
			log.Fatal(err)
		}

		// Debug print before normalization.
		fmt.Println("First mixed input (before normalization):", mixedInputs[0])
		fmt.Println("First mixed target (before normalization):", reshapedOutput[0])
		fmt.Printf("Mixed data samples: %d, Mixed targets samples: %d\n", len(mixedInputs), len(reshapedOutput))

		// Normalize mixed counting data using the same maxValue.
		for i := range mixedInputs {
			for j := range mixedInputs[i] {
				mixedInputs[i][j] /= maxValue
			}
		}
		for i := range reshapedOutput {
			reshapedOutput[i][0] /= maxValue
		}

		// Debug print after normalization.
		fmt.Println("First mixed input (after normalization):", mixedInputs[0])
		fmt.Println("First mixed target (after normalization):", reshapedOutput[0])

		fmt.Println("Training new counting model with mixed data...")
		countModel.Train(mixedInputs, reshapedOutput)
		fmt.Println("Training completed on mixed data.")

		// Evaluation on test counting data.
		var predictions []float64
		var actuals []float64
		for i := 1001; i <= 2020; i++ {
			testInput := []float64{float64(i), float64(i + 1), float64(i + 2), float64(i + 3), float64(i + 4)}
			for j := range testInput {
				testInput[j] /= maxValue
			}
			pred := countModel.Predict(testInput)
			pred[0] *= maxValue
			predictions = append(predictions, pred[0])
			actuals = append(actuals, float64(i+5))
		}
		mse := 0.0
		for i := range predictions {
			diff := predictions[i] - actuals[i]
			mse += diff * diff
		}
		mse /= float64(len(predictions))
		rmse := math.Sqrt(mse)
		countModel.RMSE = rmse
		fmt.Printf("Evaluation RMSE for counting: %.6f\n", rmse)
		rmseThreshold := 150.0
		if rmse < rmseThreshold {
			fmt.Println("RMSE acceptable. Saving counting model to database...")
			if err := countModel.SaveModel(dbConn); err != nil {
				fmt.Printf("Error saving model: %v\n", err)
			} else {
				fmt.Println("Counting model saved successfully.")
			}
		} else {
			fmt.Println("RMSE too high. Model not saved.")
		}

	case "countpredict":
		if *modelID <= 0 {
			fmt.Println("Please provide a valid model ID using -id flag for counting prediction.")
			os.Exit(1)
		}
		if *inputStr == "" {
			fmt.Println("Please provide a comma-separated list of 5 numbers using -input flag.")
			os.Exit(1)
		}
		nums, err := parseInput(*inputStr)
		if err != nil {
			fmt.Printf("Error parsing input: %v\n", err)
			os.Exit(1)
		}
		if len(nums) != 5 {
			fmt.Println("Please provide exactly 5 numbers for counting prediction.")
			os.Exit(1)
		}
		loadedModel, err := elm.LoadModel(dbConn, *modelID)
		if err != nil {
			fmt.Printf("Error loading counting model: %v\n", err)
			os.Exit(1)
		}
		maxValue := float64(6000 + 5)
		normalizedInput := make([]float64, len(nums))
		for i, v := range nums {
			normalizedInput[i] = v / maxValue
		}
		fmt.Println("Predicted next five numbers:")
		currentInput := normalizedInput
		for i := 0; i < 5; i++ {
			pred := loadedModel.Predict(currentInput)
			nextNum := pred[0] * maxValue
			fmt.Printf("%d: %.4f\n", i+1, nextNum)
			currentInput = append(currentInput[1:], pred[0])
		}

	default:
		fmt.Println("Invalid mode. Use -mode with one of: addnew, addpredict, countnew, countpredict, combineTech, protein, list, or drop")
	}
}

// argMax returns the index of the maximum value in the slice.
func argMax(slice []float64) int {
	maxIdx := 0
	maxVal := slice[0]
	for i, v := range slice {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}
