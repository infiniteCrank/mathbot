package main

import (
	"bufio"
	"database/sql"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/infiniteCrank/mathbot/db"
	"github.com/infiniteCrank/mathbot/elm"
	_ "github.com/lib/pq"
)

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

func main() {
	// Mode flag now supports: addnew, addpredict, countnew, countpredict, list, drop.
	mode := flag.String("mode", "addnew", "Mode: 'addnew', 'addpredict', 'countnew', 'countpredict', 'list', or 'drop'")
	// For loading/predicting, provide the model ID.
	modelID := flag.Int("id", 0, "ID of the model to load (used with addpredict, countpredict, or list)")
	// For prediction modes, provide input numbers.
	inputStr := flag.String("input", "", "Comma-separated list of numbers as input (2 for addpredict, 5 for countpredict)")
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

	// Create tables if they do not exist (unless dropping was performed).
	if err := createTables(dbConn); err != nil {
		fmt.Printf("Error creating tables: %v\n", err)
		os.Exit(1)
	}

	// Handle remaining modes.
	switch *mode {
	case "list":
		listModels(dbConn)
	// ------------------ Addition Modes ------------------
	case "addnew":
		// Train a new addition model.
		numSamples := 1000
		var trainingInputs [][]float64
		var trainingTargets [][]float64
		inputMax := 100.0  // Inputs in [0,100]
		outputMax := 200.0 // Sum in [0,200]
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
		// Predict addition result.
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
		// Train a new counting model.
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
		for i := range trainingInputs {
			for j := range trainingInputs[i] {
				trainingInputs[i][j] /= maxValue
			}
		}
		for i := range trainingTargets {
			trainingTargets[i][0] /= maxValue
		}
		fmt.Printf("Generated %d training examples for counting.\n", len(trainingInputs))
		countModel := elm.NewELM(5, 50, 1, 0, 0.001)
		countModel.ModelType = "counting"
		fmt.Println("Training new counting model...")
		countModel.Train(trainingInputs, trainingTargets)
		fmt.Println("Training completed.")
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
		// Predict the next five numbers using a counting model.
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
		// Use the same normalization constant as during training.
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
		fmt.Println("Invalid mode. Use -mode with one of: addnew, addpredict, countnew, countpredict, list, or drop")
	}
}
