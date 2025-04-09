# mathbot
Julian's neuro networked trained in math 

## To train a new addition model:
```go run main.go -mode=addnew```
This will train an addition model, assign its model type as "addition", and save it (if RMSE is acceptable).

## To predict an addition (e.g., the sum of 12 and 34) using a saved model with ID 1:
```go run main.go -mode=addpredict -id=1 -input="12,34"```

## To train a new counting model:
```go run main.go -mode=countnew```
This will train a counting model (using 5-number sequences), assign its model type as "counting", and save it if its performance is acceptable.

## To predict the next five numbers from the sequence "1001,1002,1003,1004,1005" using a saved counting model with ID 2:
```go run main.go -mode=countpredict -id=2 -input="1001,1002,1003,1004,1005"```

## To list all saved models with their types:
```go run main.go -mode=list```

## To drop all tables (and then start fresh):
```go run main.go -mode=drop```

## To retrain a counting model
```go run main.go -mode=countingTrain -id=1```

## To retrain a add model
```go run main.go -mode=addTrain -id=1```

## To export a model from the database to a JSON file, you would use the command line as follows:
```go run main.go -mode=export -id=1```

## To import a model from a JSON file back into your application and optionally save it in the database, use the following command
```go run main.go -mode=import -filename=model_1.json```