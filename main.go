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

	neuralnetwork "github.com/infiniteCrank/mathbot/NeuralNetwork"
	qaelm "github.com/infiniteCrank/mathbot/agent"
	"github.com/infiniteCrank/mathbot/db"
	"github.com/infiniteCrank/mathbot/elm"
	"github.com/infiniteCrank/mathbot/fileLoader"
	_ "github.com/lib/pq"
)

// rng is the random number generator for reproducible sequences
var rng *rand.Rand

///////////////////////////
//GENERATE TRAINING DATA //
///////////////////////////

// makeTestSeq creates an arithmetic sequence of given length and step
func makeTestSeq(start float64, step float64, length int) []float64 {
	seq := make([]float64, length)
	for i := 0; i < length; i++ {
		seq[i] = start + float64(i)*step
	}
	return seq
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

// generateArithData creates nSamples of arithmetic sequences with random start and step
func generateArithData(nSamples, seqLen, predLen int) ([][]float64, [][]float64) {
	inputs := make([][]float64, nSamples)
	targets := make([][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		start := rng.Float64()*10 - 5 // random start in [-5,5)
		step := rng.Float64()*4 - 2   // random step in [-2,2)
		seq := make([]float64, seqLen+predLen)
		for j := range seq {
			seq[j] = start + float64(j)*step
		}
		inputs[i] = makeSeqFeatures(seq[:seqLen])
		targets[i] = seq[seqLen:]
	}
	return inputs, targets
}

// makeSeqFeatures constructs feature vector: raw values, differences, positions, quadratic
func makeSeqFeatures(x []float64) []float64 {
	n := len(x)
	// allocate capacity: n raw + (n-1) diffs + n positions + n squares
	feat := make([]float64, 0, n*4-1)
	// raw values
	feat = append(feat, x...)
	// pairwise diffs
	for i := 1; i < n; i++ {
		feat = append(feat, x[i]-x[i-1])
	}
	// positions 1..n
	for i := 0; i < n; i++ {
		feat = append(feat, float64(i+1))
	}
	// quadratic
	for _, v := range x {
		feat = append(feat, v*v)
	}
	return feat
}

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

// argMax returns the index of the maximum value in a slice.
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
		var regularization, rmse float64
		var modelType string
		if err := rows.Scan(&id, &inputSize, &hiddenSize, &outputSize, &activation, &regularization, &modelType, &rmse); err != nil {
			return fmt.Errorf("error scanning model row: %v", err)
		}
		fmt.Printf("ID: %d | Type: %s | Input Size: %d | Hidden Size: %d | Output Size: %d | Activation: %d | Regularization: %.4f | RMSE: %.4f\n",
			id, modelType, inputSize, hiddenSize, outputSize, activation, regularization, rmse)
	}
	return nil
}

//////////////////////////
// Protein Data Section //
//////////////////////////

// ProteinSample represents a protein sample with its amino acid sequence and corresponding secondary structure.
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

// encodeAminoAcid returns a one-hot vector (length 20) for the provided amino acid.
func encodeAminoAcid(aa rune) []float64 {
	vec := make([]float64, 20)
	if idx, ok := aminoAcidMap[aa]; ok {
		vec[idx] = 1.0
	}
	return vec
}

// encodeWindow creates a flattened vector representing a sliding window of amino acids.
func encodeWindow(sequence string, pos, windowSize int) []float64 {
	half := windowSize / 2
	encoded := make([]float64, 0, windowSize*20)
	seqRunes := []rune(sequence)
	seqLen := len(seqRunes)
	for i := pos - half; i <= pos+half; i++ {
		if i < 0 || i >= seqLen {
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
	'C': {0, 0, 1}, // Coil or other
}

// generateProteinDataset creates protein training examples using a sliding window approach.
func generateProteinDataset(windowSize int) ([][]float64, [][]float64) {
	dataset := []ProteinSample{
		{
			Sequence:  "ACDEFGHIKLMNPQRSTVWY",
			Structure: "HHHHEEEECCCCCHHHHEEE",
		},
		{
			Sequence:  "MKTIIALSYIFCLVFADYKDDDDK",
			Structure: "CCCCCHHHHCCCCEEEECCCCCCC",
		},
	}

	var inputs [][]float64
	var outputs [][]float64
	for _, sample := range dataset {
		seqRunes := []rune(sample.Sequence)
		structRunes := []rune(sample.Structure)
		if len(seqRunes) != len(structRunes) {
			fmt.Printf("Warning: sequence and structure lengths do not match for sample %v\n", sample)
			continue
		}
		for pos := 0; pos < len(seqRunes); pos++ {
			inputVec := encodeWindow(sample.Sequence, pos, windowSize)
			centerStructure := structRunes[pos]
			label, ok := structureMap[centerStructure]
			if !ok {
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

	// Supported modes: addnew, addpredict, addTrain, countnew, countpredict, countingTrain, combineTech, protein, list, drop.
	mode := flag.String("mode", "addnew", "Mode: 'addnew','addpredict','addTrain','countnew','countpredict','countingTrain','combineTech','protein','list','drop','sequence'")
	modelID := flag.Int("id", 0, "Model ID for load/predict/retrain (used with addpredict, addTrain, countpredict, countingTrain, or list)")
	inputStr := flag.String("input", "", "Comma-separated list of numbers as input (used in prediction modes)")
	filename := flag.String("filename", "", "Path to the model file for import")
	// new flags for one‑off teaching
	learnQ := flag.String("learnQ", "", "Question to teach the agent (one‑off)")
	learnA := flag.String("learnA", "", "Correct answer for learnQ (one‑off)")
	agentSaveInterval := flag.Int("agent-save-interval", 3, "Save agent model every N feedback updates (agent mode)")
	validationSplit := flag.Float64("validation-split", 0.2, "Fraction of data to use for validation")
	nSamples := flag.Int("n", 10000, "number of training samples")
	seqLen := flag.Int("seq", 5, "length of input sequence")
	predLen := flag.Int("pred", 5, "length of prediction sequence")
	hiddenSize := flag.Int("h", 1000, "number of hidden neurons")
	lambda := flag.Float64("lambda", 1e-4, "ridge regularization parameter")
	seed := flag.Int64("seed", 42, "random seed for reproducibility")

	flag.Parse()
	// Initialize a local RNG instead of using the global one
	rng = rand.New(rand.NewSource(*seed))

	dbConn := db.ConnectDB()
	defer dbConn.Close()

	// If both learnQ and learnA are provided, teach and exit immediately.
	if *mode == "agent" && *learnQ != "" && *learnA != "" {
		// Load and initialize agent using the provided Q&A pairs.
		docs, err := fileLoader.LoadMarkdownFiles("corpus")
		if err != nil || len(docs) == 0 {
			log.Fatalf("Failed to load corpus for teaching: %v", err)
		}

		// Parse the Q/A pairs from the loaded documents.
		qas := qaelm.ParseMarkdownQA(docs)
		if len(qas) == 0 {
			log.Fatalf("No Q/A pairs found in the loaded documents.")
		}

		// Initialize the agent with parameters (update values as needed).
		agent, err := qaelm.NewQAELMAgent(qas, 32, 0, 0.001, *validationSplit)
		if err != nil {
			log.Fatalf("Failed to init agent: %v", err)
		}

		// Teach the agent with the new Q/A pair.
		if err := agent.Learn(qaelm.QA{Question: *learnQ, Answer: *learnA}); err != nil {
			log.Fatalf("Learn error: %v", err)
		}

		// Save the updated model to the database.
		if err := agent.Model.SaveModel(dbConn); err != nil {
			log.Fatalf("SaveModel error: %v", err)
		}

		fmt.Printf("Taught Q=%q, A=%q and saved model successfully.\n", *learnQ, *learnA)
		return
	}

	// Handle drop mode first.
	if *mode == "drop" {
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("Are you sure you want to drop all elm tables? (y/n): ")
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

	// Ensure tables exist.
	if err := createTables(dbConn); err != nil {
		fmt.Printf("Error creating tables: %v\n", err)
		os.Exit(1)
	}

	switch *mode {

	case "protein":
		windowSize := 15
		inputs, outputs := generateProteinDataset(windowSize)
		fmt.Printf("Generated %d protein training examples.\n", len(inputs))

		inputSize := windowSize * 20 // 20 is the size for one-hot encoding of amino acids
		hiddenSize := 50
		outputSize := 3 // Output for the three classes (Helix, Sheet, Coil)
		activation := 0 // Choose appropriate activation function (e.g., 0 for Sigmoid)
		regularization := 0.01

		// Create the protein model
		proteinModel := elm.NewELM(inputSize, hiddenSize, outputSize, activation, regularization)
		proteinModel.ModelType = "protein_structure"

		fmt.Println("Training protein structure prediction model...")
		// Train the model using the training dataset
		proteinModel.Train(inputs, outputs, nil, nil) // For now, we pass nil since no validation is specified
		fmt.Println("Training completed.")

		correct := 0
		for i, input := range inputs {
			pred := proteinModel.Predict(input)
			if argMax(pred) == argMax(outputs[i]) {
				correct++
			}
		}
		accuracy := float64(correct) / float64(len(inputs)) * 100.0
		fmt.Printf("Training accuracy: %.2f%%\n", accuracy)

		if accuracy > 70.0 {
			fmt.Println("Accuracy acceptable. Saving protein model...")
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

		// Generate training samples.
		for i := 0; i < numSamples; i++ {
			a := rand.Float64() * inputMax
			b := rand.Float64() * inputMax
			// Normalize inputs and outputs
			trainingInputs = append(trainingInputs, []float64{a / inputMax, b / inputMax})
			trainingTargets = append(trainingTargets, []float64{(a + b) / outputMax}) // Normalize target
		}
		fmt.Printf("Generated %d training examples for addition.\n", numSamples)

		// Create and train the addition model.
		addModel := elm.NewELM(2, 10, 1, 0, 0.001)
		addModel.ModelType = "addition" // Set model type for identification
		fmt.Println("Training new addition model...")

		// Train the model using the training data
		addModel.Train(trainingInputs, trainingTargets, nil, nil) // No validation data for now
		fmt.Println("Training completed.")

		// Evaluation logic
		var mseSum float64
		for i := 0; i < 100; i++ { // Evaluate over 100 samples
			a := rand.Float64() * inputMax
			b := rand.Float64() * inputMax
			testInput := []float64{a / inputMax, b / inputMax} // Normalize test input
			pred := addModel.Predict(testInput)
			predictedSum := pred[0] * outputMax // Denormalize output
			trueSum := a + b
			diff := predictedSum - trueSum
			mseSum += diff * diff // Calculate mean squared error
		}
		rmse := math.Sqrt(mseSum / 100) // Calculate root mean square error
		addModel.RMSE = rmse
		fmt.Printf("Evaluation RMSE for addition: %.6f\n", rmse)

		// Save the model if RMSE is within acceptable limits
		if rmse < 5.0 {
			fmt.Println("RMSE acceptable. Saving addition model...")
			if err := addModel.SaveModel(dbConn); err != nil {
				fmt.Printf("Error saving addition model: %v\n", err)
			} else {
				fmt.Println("Addition model saved successfully.")
			}
		} else {
			fmt.Println("RMSE too high. Model not saved.")
		}

	case "addpredict":
		if *modelID <= 0 {
			fmt.Println("Please provide a valid model ID for addition prediction using -id flag.")
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

	case "addTrain":
		if *modelID <= 0 {
			fmt.Println("Please provide a valid model ID for addition retraining using -id flag.")
			os.Exit(1)
		}
		loadedModel, err := elm.LoadModel(dbConn, *modelID)
		if err != nil {
			fmt.Printf("Error loading addition model: %v\n", err)
			os.Exit(1)
		}
		numSamples := 1000
		var trainingInputs [][]float64
		var trainingTargets [][]float64
		inputMax := 100.0
		outputMax := 200.0

		// Generate training data
		for i := 0; i < numSamples; i++ {
			a := rand.Float64() * inputMax
			b := rand.Float64() * inputMax
			trainingInputs = append(trainingInputs, []float64{a / inputMax, b / inputMax})
			trainingTargets = append(trainingTargets, []float64{(a + b) / outputMax})
		}
		fmt.Printf("Generated %d new training examples for addition retraining.\n", numSamples)

		// Create a validation dataset (e.g., using the last 200 examples)
		var valInputs [][]float64
		var valTargets [][]float64
		for i := numSamples; i < numSamples+200; i++ {
			a := rand.Float64() * inputMax
			b := rand.Float64() * inputMax
			valInputs = append(valInputs, []float64{a / inputMax, b / inputMax})
			valTargets = append(valTargets, []float64{(a + b) / outputMax})
		}
		fmt.Printf("Generated %d validation examples for addition retraining.\n", len(valInputs))

		fmt.Println("Retraining addition model with new data...")
		// Pass training and validation sets along with patience to Train
		loadedModel.Train(trainingInputs, trainingTargets, valInputs, valTargets)
		fmt.Println("Retraining completed.")

		// Evaluation of the model after retraining
		var mseSumAT float64
		for i := 0; i < 100; i++ {
			a := rand.Float64() * inputMax
			b := rand.Float64() * inputMax
			testInput := []float64{a / inputMax, b / inputMax}
			pred := loadedModel.Predict(testInput)
			predictedSum := pred[0] * outputMax
			trueSum := a + b
			diff := predictedSum - trueSum
			mseSumAT += diff * diff
		}
		rmseAT := math.Sqrt(mseSumAT / 100)
		loadedModel.RMSE = rmseAT
		fmt.Printf("Evaluation RMSE after addition retraining: %.6f\n", rmseAT)
		if rmseAT < 5.0 {
			fmt.Println("RMSE acceptable. Saving updated addition model...")
			if err := loadedModel.SaveModel(dbConn); err != nil {
				fmt.Printf("Error saving updated addition model: %v\n", err)
			} else {
				fmt.Println("Updated addition model saved successfully.")
			}
		} else {
			fmt.Println("RMSE too high. Updated addition model not saved.")
		}
	// ------------------ Counting Modes ------------------
	case "countnew":
		startNum := 1
		endNum := 6000
		var trainingInputs [][]float64
		var trainingTargets [][]float64
		for i := startNum; i <= endNum-5; i++ {
			seq := []float64{
				float64(i),
				float64(i + 1),
				float64(i + 2),
				float64(i + 3),
				float64(i + 4),
			}
			trainingInputs = append(trainingInputs, seq)
			trainingTargets = append(trainingTargets, []float64{float64(i + 5)})
		}
		maxValue := float64(endNum + 5)
		// Normalize training data
		for i := range trainingInputs {
			for j := range trainingInputs[i] {
				trainingInputs[i][j] /= maxValue
			}
		}
		for i := range trainingTargets {
			trainingTargets[i][0] /= maxValue
		}
		fmt.Printf("Generated %d training examples for counting.\n", len(trainingInputs))

		// Create a validation dataset (e.g., using the last 100 sequential examples)
		var valInputs [][]float64
		var valTargets [][]float64
		for i := endNum - 100; i <= endNum-5; i++ {
			seq := []float64{
				float64(i),
				float64(i + 1),
				float64(i + 2),
				float64(i + 3),
				float64(i + 4),
			}
			valInputs = append(valInputs, seq)
			valTargets = append(valTargets, []float64{float64(i + 5)})
		}
		// Normalize validation data
		for i := range valInputs {
			for j := range valInputs[i] {
				valInputs[i][j] /= maxValue
			}
		}
		for i := range valTargets {
			valTargets[i][0] /= maxValue
		}

		countModel := elm.NewELM(5, 50, 1, 0, 0.001)
		countModel.ModelType = "counting"
		fmt.Println("Training new counting model on sequential data...")

		// Pass validation inputs and targets along with patience to Train
		countModel.Train(trainingInputs, trainingTargets, valInputs, valTargets)
		fmt.Println("Training completed on sequential data.")

		var predictions []float64
		var actuals []float64
		for i := 1001; i <= 2020; i++ {
			testInput := []float64{
				float64(i),
				float64(i + 1),
				float64(i + 2),
				float64(i + 3),
				float64(i + 4),
			}
			// Normalize input
			for j := range testInput {
				testInput[j] /= maxValue
			}
			pred := countModel.Predict(testInput)
			predictedNum := pred[0] * maxValue
			predictions = append(predictions, predictedNum)
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
		if rmse < 1 {
			fmt.Println("RMSE acceptable. Saving counting model...")
			if err := countModel.SaveModel(dbConn); err != nil {
				fmt.Printf("Error saving counting model: %v\n", err)
			} else {
				fmt.Println("Counting model saved successfully.")
			}
		} else {
			fmt.Println("RMSE too high. Counting model not saved.")
		}

	case "countpredict":
		if *modelID <= 0 {
			fmt.Println("Please provide a valid model ID for counting prediction using -id flag.")
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

	case "countingTrain":
		// This mode retrains an existing counting model by merging sequential and mixed data.
		if *modelID <= 0 {
			fmt.Println("Please provide a valid model ID for counting retraining using -id flag.")
			os.Exit(1)
		}
		loadedModel, err := elm.LoadModel(dbConn, *modelID)
		if err != nil {
			fmt.Printf("Error loading counting model: %v\n", err)
			os.Exit(1)
		}
		// Generate sequential counting data.
		startNum := 1
		endNum := 6000
		var seqInputs [][]float64
		var seqTargets [][]float64
		for i := startNum; i <= endNum-5; i++ {
			seq := []float64{
				float64(i),
				float64(i + 1),
				float64(i + 2),
				float64(i + 3),
				float64(i + 4),
			}
			seqInputs = append(seqInputs, seq)
			seqTargets = append(seqTargets, []float64{float64(i + 5)})
		}
		maxValue := float64(endNum + 5)
		// Normalize sequential data.
		for i := range seqInputs {
			for j := range seqInputs[i] {
				seqInputs[i][j] /= maxValue
			}
		}
		for i := range seqTargets {
			seqTargets[i][0] /= maxValue
		}
		// Evaluate on test set.
		var predictions []float64
		var actuals []float64
		for i := 1001; i <= 2020; i++ {
			testInput := []float64{
				float64(i),
				float64(i + 1),
				float64(i + 2),
				float64(i + 3),
				float64(i + 4),
			}
			for j := range testInput {
				testInput[j] /= maxValue
			}
			pred := loadedModel.Predict(testInput)
			pred[0] *= maxValue
			predictions = append(predictions, pred[0])
			actuals = append(actuals, float64(i+5))
		}
		mseCT := 0.0
		for i := range predictions {
			diff := predictions[i] - actuals[i]
			mseCT += diff * diff
		}
		mseCT /= float64(len(predictions))
		rmseCT := math.Sqrt(mseCT)
		loadedModel.RMSE = rmseCT
		fmt.Printf("Evaluation RMSE after counting retraining: %.6f\n", rmseCT)
		if rmseCT < loadedModel.RMSE {
			fmt.Println("RMSE acceptable. Saving updated counting model...")
			if err := loadedModel.SaveModel(dbConn); err != nil {
				fmt.Printf("Error saving updated counting model: %v\n", err)
			} else {
				fmt.Println("Updated counting model saved successfully.")
			}
		} else {
			fmt.Println("RMSE too high. Updated counting model not saved.")
		}
	case "export":
		if *modelID <= 0 {
			fmt.Println("Please provide a valid model ID for exporting using -id flag.")
			os.Exit(1)
		}
		loadedModel, err := elm.LoadModel(dbConn, *modelID)
		if err != nil {
			fmt.Printf("Error loading model for export: %v\n", err)
			os.Exit(1)
		}
		filename := fmt.Sprintf("model_%d.json", *modelID)
		if err := loadedModel.ExportToFile(filename); err != nil {
			fmt.Printf("Error exporting model to file: %v\n", err)
		} else {
			fmt.Printf("Model exported successfully to %s\n", filename)
		}

	case "import":
		if *filename == "" {
			fmt.Println("Please provide the path to the model file using -filename flag.")
			os.Exit(1)
		}
		importedModel, err := elm.ImportModelFromFile(*filename)
		if err != nil {
			fmt.Printf("Error importing model from file: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("Model imported successfully.")

		if err := importedModel.SaveModel(dbConn); err != nil {
			fmt.Printf("Error saving imported model to database: %v\n", err)
		} else {
			fmt.Println("Model saved to database successfully.")
		}
	case "agent":
		// Load corpus and parse QA
		docs, err := fileLoader.LoadMarkdownFiles("corpus")
		if err != nil {
			log.Fatalf("Failed to load markdown: %v", err)
		}
		if len(docs) == 0 {
			log.Fatalf("No markdown files in 'corpus'")
		}
		fmt.Printf("Loaded %d markdown files\n", len(docs))

		qas := qaelm.ParseMarkdownQA(docs)
		if len(qas) == 0 {
			log.Fatalf("No Q/A pairs found. Ensure '## Q:' headings.")
		}
		fmt.Printf("Extracted %d Q/A pairs\n", len(qas))

		// Train or load agent
		hiddenSize := 50
		activation := 0
		lambda := 0.000001 // Regularization parameter
		agent, err := qaelm.NewQAELMAgent(qas, hiddenSize, activation, lambda, *validationSplit)
		if err != nil {
			log.Fatalf("Failed to init agent: %v", err)
		}
		fmt.Println("Agent initialized.")

		// If loading an existing model
		if *modelID > 0 {
			loaded, err := elm.LoadModel(dbConn, *modelID)
			if err != nil {
				log.Fatalf("Failed to load agent ELM weights: %v", err)
			}
			agent.Model = loaded // override ELM inside agent
			fmt.Printf("Loaded agent model ID %d from database.\n", *modelID)
		}

		// Interactive loop with periodic save
		scanner := bufio.NewScanner(os.Stdin)
		updateCount := 0
		for {
			fmt.Print("Enter question ('exit' to quit): ")
			if !scanner.Scan() {
				break
			}
			q := scanner.Text()
			if strings.EqualFold(q, "exit") {
				fmt.Println("Goodbye!")
				break
			}

			// Ask
			ans, err := agent.Ask(q)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
			} else {
				fmt.Printf("Answer: %s\n", ans)
			}

			// Feedback & learn (with panic‑protection)
			fmt.Print("If incorrect, enter correct answer (or blank): ")
			if !scanner.Scan() {
				break
			}
			fb := scanner.Text()
			if fb != "" && !strings.EqualFold(fb, ans) {
				// 1) Record user correction in corpus file
				f, err := os.OpenFile("corpus/feedback.md", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
				if err == nil {
					fmt.Fprintf(f, "## Q: %s\n A: %s\n\n", q, fb)
					f.Close()
				}

				// 2) Append to in-memory corpus
				agent.Corpus = append(agent.Corpus, qaelm.QA{Question: q, Answer: fb})

				// 3) Fully retrain the agent on the expanded corpus
				if err := agent.Retrain(*validationSplit); err != nil {
					fmt.Printf("⚠️  Retrain error: %v\n", err)
				} else {
					fmt.Println("Agent fully retrained with new feedback.")
				}

				// 4) Persist the updated model
				if err := agent.Model.SaveModel(dbConn); err != nil {
					fmt.Printf("SaveModel error: %v\n", err)
				} else {
					fmt.Println("Agent model saved after retrain.")
				}
			}
		}

		// On exit, final save if any updates
		if *agentSaveInterval > 0 && updateCount > 0 {
			err := agent.Model.SaveModel(dbConn)
			if err != nil {
				fmt.Printf("Final SaveModel error: %v\n", err)
			} else {
				fmt.Println("Final agent model saved successfully.")
			}
		}
	case "countBySteps":
		// Generate and split data
		allX, allY := generateArithData(*nSamples, *seqLen, *predLen)
		valSize := *nSamples / 5
		valX, valY := allX[:valSize], allY[:valSize]
		trainX, trainY := allX[valSize:], allY[valSize:]

		// Train ELM
		inputSize := len(trainX[0])
		elmModel := elm.NewELM(inputSize, *hiddenSize, *predLen, 1, *lambda)
		elmModel.Train(trainX, trainY, valX, valY)

		// Compute validation RMSE
		var sum float64
		for i := range valX {
			predVec := elmModel.Predict(valX[i])
			for j := range predVec {
				diff := predVec[j] - valY[i][j]
				sum += diff * diff
			}
		}
		rmse := math.Sqrt(sum / float64(len(valX)*len(valY[0])))
		fmt.Printf("Validation RMSE: %.4f\n", rmse)

		// Save model if RMSE acceptable
		if rmse <= 1 {
			elmModel.ModelType = "counting"
			if err := elmModel.SaveModel(dbConn); err != nil {
				fmt.Fprintf(os.Stderr, "Failed to save model: %v\n", err)
			} else {
				fmt.Println("Model saved as 'counting' type.")
			}
		} else {
			fmt.Println("RMSE too high. Model not saved.")
		}

		// Test predictions
		seqLenVal := *seqLen
		tests := [][]float64{
			makeTestSeq(1, 1, seqLenVal),
			makeTestSeq(2, 2, seqLenVal),
		}
		for _, seq := range tests {
			feat := makeSeqFeatures(seq)
			pred := elmModel.Predict(feat)
			fmt.Printf("Input: %v\nPred:  ", seq)
			for _, v := range pred {
				fmt.Printf("%.2f ", v)
			}
			fmt.Println("\n---")
		}
		// after your validation & (optional) SaveModel logic:
		testGeneralization(elmModel, *seqLen, *predLen, 1000)

		gridTest(elmModel, *seqLen, *predLen,
			-5, 5, // start range
			-2, 2, // step range
			20, 20) // 20×20 grid

		randomTest(elmModel, *seqLen, *predLen, 2000, -100, 100, -10, 10)

		autoregTest(elmModel, *seqLen, *predLen, 1000, -5, 5, -2, 2)
		autoregTest(elmModel, *seqLen, *predLen, 1000, -100, 100, -10, 10)
	case "neuroNetwork":
		rand.Seed(time.Now().UnixNano())
		seqLen, predLen := 5, 5
		nSamples := 1000
		trainFrac := 0.8

		// Generate and split data
		X, Y := generateArithData(nSamples, seqLen, predLen)
		nTrain := int(trainFrac * float64(nSamples))
		trainX, trainY := X[:nTrain], Y[:nTrain]
		valX, valY := X[nTrain:], Y[nTrain:]

		// Fit scalers & normalize
		xMeans, xStds := neuralnetwork.FitScaler(trainX)
		yMeans, yStds := neuralnetwork.FitScaler(trainY)
		trainX = neuralnetwork.Normalize(trainX, xMeans, xStds)
		valX = neuralnetwork.Normalize(valX, xMeans, xStds)
		trainY = neuralnetwork.Normalize(trainY, yMeans, yStds)
		valY = neuralnetwork.Normalize(valY, yMeans, yStds)

		// Build, train, and validate
		inputDim := len(trainX[0])
		nn := neuralnetwork.NewNetwork([]int{inputDim, 64, 64, predLen}, 0.01, 100, 32, 10)
		nn.Train(trainX, trainY, valX, valY)

		// Test on a new 1,3,5,7,9 sequence
		testSeq := []float64{1, 3, 5, 7, 9}
		testFeat := makeSeqFeatures(testSeq)
		tn := neuralnetwork.Normalize([][]float64{testFeat}, xMeans, xStds)[0]
		predNorm := nn.Predict(tn)
		pred := neuralnetwork.Denormalize(predNorm, yMeans, yStds)

		fmt.Printf("Input: %v\n", testSeq)
		fmt.Printf("Predicted next %d: %v\n", predLen, pred)
	default:
		fmt.Println("Invalid mode. Use -mode with one of: addnew, addpredict, addTrain, countnew, countpredict, countingTrain, combineTech, protein, list, or drop")
	}

}

// testGeneralization runs out‑of‑sample arithmetic‐sequence tests
// to measure how well the ELM learned the “by‑steps” rule.
func testGeneralization(
	model *elm.ELM,
	seqLen, predLen, nTests int,
) {
	type caseDef struct{ start, step float64 }
	var cases []caseDef

	// 1. Build nTests random (start, step) pairs from a wide range
	for i := 0; i < nTests; i++ {
		cases = append(cases, caseDef{
			start: rand.Float64()*200 - 100, // from –100 to +100
			step:  rand.Float64()*20 - 10,   // from –10 to +10
		})
	}

	var sumSq, sumAbs float64
	var total int

	for _, c := range cases {
		// build the full truth sequence
		seq := make([]float64, seqLen+predLen)
		for i := range seq {
			seq[i] = c.start + float64(i)*c.step
		}

		// 2. Single‐shot prediction (the same as your existing tests):
		//    build features from the first seqLen, predict predLen in one go
		pred := model.Predict(makeSeqFeatures(seq[:seqLen]))

		// accumulate errors
		for j := 0; j < predLen; j++ {
			err := pred[j] - seq[seqLen+j]
			sumSq += err * err
			sumAbs += math.Abs(err)
			total++
		}

		// 3. (Optional) autoregressive multi‐step forecasting:
		//    uncomment if you want to see how errors grow when
		//    feeding predictions back in step by step.

		window := append([]float64{}, seq[:seqLen]...)
		for j := 0; j < predLen; j++ {
			p := model.Predict(makeSeqFeatures(window))[0]
			err := p - seq[seqLen+j]
			sumSq += err * err
			sumAbs += math.Abs(err)
			total++

			// slide window
			window = append(window[1:], p)
		}

	}

	rmse := math.Sqrt(sumSq / float64(total))
	mae := sumAbs / float64(total)
	fmt.Printf(
		"Generalization over %d cases (seqLen=%d, predLen=%d): RMSE=%.4f  MAE=%.4f\n",
		total/predLen, seqLen, predLen, rmse, mae,
	)
}

// gridTest runs a dense grid over [startMin,startMax]×[stepMin,stepMax]
func gridTest(
	model *elm.ELM,
	seqLen, predLen int,
	startMin, startMax float64,
	stepMin, stepMax float64,
	nStart, nStep int,
) {
	var sumSq, sumAbs float64
	var count int

	for i := 0; i < nStart; i++ {
		start := startMin + (startMax-startMin)*float64(i)/float64(nStart-1)
		for j := 0; j < nStep; j++ {
			step := stepMin + (stepMax-stepMin)*float64(j)/float64(nStep-1)

			// build the truth sequence
			seq := make([]float64, seqLen+predLen)
			for k := range seq {
				seq[k] = start + float64(k)*step
			}

			// single‑shot predict
			pred := model.Predict(makeSeqFeatures(seq[:seqLen]))
			for k := 0; k < predLen; k++ {
				err := pred[k] - seq[seqLen+k]
				sumSq += err * err
				sumAbs += math.Abs(err)
				count++
			}
		}
	}

	rmse := math.Sqrt(sumSq / float64(count))
	mae := sumAbs / float64(count)
	fmt.Printf("Inside‑grid test (starts∈[%.1f,%.1f], steps∈[%.1f,%.1f]): RMSE=%.4f  MAE=%.4f\n",
		startMin, startMax, stepMin, stepMax, rmse, mae)
}

func randomTest(
	model *elm.ELM,
	seqLen, predLen, nTests int,
	startMin, startMax, stepMin, stepMax float64,
) {
	var sumSq, sumAbs float64
	for i := 0; i < nTests; i++ {
		start := rand.Float64()*(startMax-startMin) + startMin
		step := rand.Float64()*(stepMax-stepMin) + stepMin

		seq := make([]float64, seqLen+predLen)
		for k := range seq {
			seq[k] = start + float64(k)*step
		}
		pred := model.Predict(makeSeqFeatures(seq[:seqLen]))

		for k := 0; k < predLen; k++ {
			err := pred[k] - seq[seqLen+k]
			sumSq += err * err
			sumAbs += math.Abs(err)
		}
	}
	count := float64(nTests * predLen)
	fmt.Printf("Random‑test (%d cases from starts∈[%.0f,%.0f], steps∈[%.0f,%.0f]): RMSE=%.4f  MAE=%.4f\n",
		nTests, startMin, startMax, stepMin, stepMax,
		math.Sqrt(sumSq/count), sumAbs/count,
	)
}

func autoregTest(
	model *elm.ELM,
	seqLen, predLen, nTests int,
	startMin, startMax, stepMin, stepMax float64,
) {
	var sumSq, sumAbs float64
	for i := 0; i < nTests; i++ {
		start := rand.Float64()*(startMax-startMin) + startMin
		step := rand.Float64()*(stepMax-stepMin) + stepMin

		// build the initial window
		window := make([]float64, seqLen)
		for k := 0; k < seqLen; k++ {
			window[k] = start + float64(k)*step
		}

		// truth for next predLen steps
		truth := make([]float64, predLen)
		for k := 0; k < predLen; k++ {
			truth[k] = start + float64(seqLen+k)*step
		}

		// roll forward
		for k := 0; k < predLen; k++ {
			pred := model.Predict(makeSeqFeatures(window))[0]
			err := pred - truth[k]
			sumSq += err * err
			sumAbs += math.Abs(err)

			// slide window
			window = append(window[1:], pred)
		}
	}
	count := float64(nTests * predLen)
	fmt.Printf("Autoregressive‑test (%d): RMSE=%.4f  MAE=%.4f\n",
		nTests,
		math.Sqrt(sumSq/count),
		sumAbs/count,
	)
}
