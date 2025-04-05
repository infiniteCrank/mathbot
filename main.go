package main

import (
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
			regularization FLOAT8
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

func main() {
	// Command line flags.
	mode := flag.String("mode", "new", "Mode: 'new' to train a new model, 'load' to load an existing model, 'predict' to run a prediction")
	modelID := flag.Int("id", 0, "ID of the model to load (used with mode=load or mode=predict)")
	inputStr := flag.String("input", "", "Comma-separated list of numbers as input (used with mode=predict)")
	flag.Parse()

	db := db.ConnectDB()

	// Create tables if they do not exist.
	if err := createTables(db); err != nil {
		fmt.Printf("Error creating tables: %v\n", err)
		os.Exit(1)
	}

	// For training mode, generate training data.
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

	// Normalization constant (same used during training)
	maxValue := float64(endNum + 5)
	for i := range trainingInputs {
		for j := range trainingInputs[i] {
			trainingInputs[i][j] /= maxValue
		}
	}
	for i := range trainingTargets {
		trainingTargets[i][0] /= maxValue
	}
	fmt.Printf("Generated %d training examples.\n", len(trainingInputs))

	switch *mode {
	case "new":
		// Train a new ELM model.
		elmModel := elm.NewELM(5, 50, 1, 0, 0.001)
		fmt.Println("Training new ELM model...")
		elmModel.Train(trainingInputs, trainingTargets)
		fmt.Println("Training completed.")

		// Evaluate the new model.
		var predictions []float64
		var actuals []float64
		for i := 1001; i <= 2020; i++ {
			testInput := []float64{float64(i), float64(i + 1), float64(i + 2), float64(i + 3), float64(i + 4)}
			for j := range testInput {
				testInput[j] /= maxValue
			}
			pred := elmModel.Predict(testInput)
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
		fmt.Printf("Evaluation Metrics:\nMSE: %.6f, RMSE: %.6f\n", mse, rmse)

		// Save model if performance is acceptable.
		rmseThreshold := 50.0
		if rmse < rmseThreshold {
			fmt.Println("RMSE is acceptable. Saving model to database...")
			if err := elmModel.SaveModel(db); err != nil {
				fmt.Printf("Error saving model: %v\n", err)
			} else {
				fmt.Println("Model saved successfully.")
			}
		} else {
			fmt.Println("RMSE is too high. Model not saved.")
		}

	case "load":
		if *modelID <= 0 {
			fmt.Println("Please provide a valid model ID using -id flag.")
			os.Exit(1)
		}
		loadedModel, err := elm.LoadModel(db, *modelID)
		if err != nil {
			fmt.Printf("Error loading model: %v\n", err)
			os.Exit(1)
		}
		// Test the loaded model on a sample.
		testInput := []float64{1001, 1002, 1003, 1004, 1005}

		for j := range testInput {
			testInput[j] /= maxValue
		}
		pred := loadedModel.Predict(testInput)
		pred[0] *= maxValue
		fmt.Printf("Loaded model (ID %d)for sample: 1001, 1002, 1003, 1004, 1005 predicted %.4f\n", *modelID, pred[0])

	case "predict":
		// In predict mode, we require a model ID and an input.
		if *modelID <= 0 {
			fmt.Println("Please provide a valid model ID using -id flag for prediction.")
			os.Exit(1)
		}
		if *inputStr == "" {
			fmt.Println("Please provide a comma-separated list of numbers as input using -input flag.")
			os.Exit(1)
		}
		inputNums, err := parseInput(*inputStr)
		if err != nil {
			fmt.Printf("Error parsing input numbers: %v\n", err)
			os.Exit(1)
		}
		if len(inputNums) == 0 {
			fmt.Println("No input numbers provided.")
			os.Exit(1)
		}

		// Load the model.
		loadedModel, err := elm.LoadModel(db, *modelID)
		if err != nil {
			fmt.Printf("Error loading model: %v\n", err)
			os.Exit(1)
		}
		// For consistency, assume the model was trained with 5-number input.
		if len(inputNums) != 5 {
			fmt.Println("Please provide exactly 5 numbers as input.")
			os.Exit(1)
		}

		// Normalize the input.
		normalizedInput := make([]float64, len(inputNums))
		for i, v := range inputNums {
			normalizedInput[i] = v / maxValue
		}

		// Iteratively predict the next 5 numbers.
		fmt.Println("Predicted next five numbers:")
		currentInput := normalizedInput
		for i := 0; i < 5; i++ {
			pred := loadedModel.Predict(currentInput)
			// Denormalize prediction.
			nextNum := pred[0] * maxValue
			fmt.Printf("%d: %.4f\n", i+1, nextNum)
			// Slide window: remove first element, append predicted (normalized) number.
			currentInput = append(currentInput[1:], pred[0])
		}

	default:
		fmt.Println("Invalid mode. Use -mode=new, -mode=load, or -mode=predict")
	}
}
