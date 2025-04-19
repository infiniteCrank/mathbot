Documentation for CLI Usage
This documentation covers how to use the command-line interface (CLI) to interact with the QAELM (Question-Answer Extreme Learning Machine) agent. The CLI provides various modes for training, predicting, and managing models, along with flags to customize behavior.

CLI Commands
Teaching a New Question and Answer

Command:
go run main.go -mode=agent -learnQ="<your_question>" -learnA="<your_answer>"
Description: Teach the agent a new question and its corresponding answer. The agent will learn this information and save the updated model to the database.
Flags:
-learnQ: The question to teach the agent (string).
-learnA: The correct answer associated with the question (string).
Running the Agent in Interactive Mode

Command:
go run main.go -mode=agent
Description: Start an interactive session with the agent where you can ask questions and receive answers.
Behavior: The program will prompt you to enter a question. You can then enter feedback for the agent to learn more.
Dropping Tables in the Database

Command:
go run main.go -mode=drop
Description: Drop all elm tables in the database. This action cannot be undone, so it will prompt for confirmation.
Behavior: You will be asked to confirm your action by typing 'y' or 'n'.
Creating Database Tables

Command:
go run main.go -mode=create
Description: This is typically called automatically on startup to ensure that the necessary tables exist.
Listing Saved Models

Command:
go run main.go -mode=list
Description: Lists all saved models in the database along with their properties such as RMSE, activation function, and input/output sizes.
Training a New Addition Model

Command:
go run main.go -mode=addnew
Description: Generates training data for an addition model and trains it.
Behavior: The model will be saved if the RMSE is below the acceptable threshold.
Making Predictions with an Addition Model

Command:
go run main.go -mode=addpredict -id=<model_id> -input="<number1>,<number2>"
Description: Use an existing model to make predictions for the addition of two numbers.
Flags:
-id: The ID of the model to use (integer).
-input: A comma-separated list of two numbers for prediction (string).
Retraining an Addition Model

Command:
go run main.go -mode=addTrain -id=<model_id>
Description: Retrain an existing addition model with new training samples to improve its accuracy.
Training a Counting Model

Command:
go run main.go -mode=countnew
Description: Generate and train a new counting model using sequential data.
Making Predictions with a Counting Model

Command:
go run main.go -mode=countpredict -id=<model_id> -input="<number1>,<number2>,<number3>,<number4>,<number5>"
Description: Use a counting model to predict the next number in a sequence.
Flags:
-id: The ID of the counting model to use (integer).
-input: A comma-separated list of five numbers (string).
Retraining a Counting Model

Command:
go run main.go -mode=countingTrain -id=<model_id>
Description: Retrain an existing counting model by aggregating new training data.
Combining Techniques for Model Training

Command:
go run main.go -mode=combineTech
Description: Train a neural network model on transformed features obtained from a sine wave dataset using an ELM.
Training a Protein Structure Model

Command:
go run main.go -mode=protein
Description: Generates a protein dataset and trains a model to predict secondary structures based on amino acid sequences.
Example Usage Scenario
Teaching the Agent a New Q/A Pair:

If you have a new question about a programming concept and its answer, you can run:

go run main.go -mode=agent -learnQ="What is a closure in Go?" -learnA="A closure is a function that captures the lexical scope of its surrounding code."
Interacting with the Agent:
After initializing, you can start interacting with the agent. Simply run:

go run main.go -mode=agent
Then, in the prompt, you can ask:

Enter question: What is a closure in Go?
Answer: A closure is a function that captures the lexical scope of its surrounding code.
If incorrect, enter correct answer (or blank to skip):
If you find the answer incorrect, you might enter feedback which allows the agent to learn:

If incorrect, enter correct answer (or blank to skip): 
If you enter a correct answer, the agent will learn from it.

General Tips
Always Confirm: When using the drop mode, always confirm your intention to delete all models or tables to avoid accidental data loss.
Monitoring Performance: Check the output logs for action success, such as model saves or training completions, to keep track of the agent's activities.
Use Meaningful Questions and Answers: Ensure that teaching the agent with learnQ and learnA consists of meaningful and informative content to enhance response accuracy in future queries.
