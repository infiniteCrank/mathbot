package agent

import (
	"errors"
	"fmt"
	"log"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/infiniteCrank/mathbot/elm"
	"github.com/infiniteCrank/mathbot/tfidf"
	"github.com/jdkato/prose/v2" // Import an NLP library
	"gonum.org/v1/gonum/mat"
)

// QA holds a question-answer pair extracted from markdown.
type QA struct {
	Question string
	Answer   string
}

// QAELMAgent wraps an ELM for QA tasks.
type QAELMAgent struct {
	Model         *elm.ELM
	terms         []string           // sorted vocabulary terms
	answers       []string           // list of unique answers
	termToIndex   map[string]int     // maps term to vector index
	answerToIndex map[string]int     // maps answer to class index
	idf           map[string]float64 // inverse document frequency from training corpus
	// Online-update matrices
	P          *mat.Dense // (H^T H + λI)^{-1}
	Beta       *mat.Dense // output weights
	hiddenSize int
	lambda     float64

	Corpus []QA // all seen QAs, including initially parsed + any

	mutex sync.Mutex
}

// NewQAELMAgent builds and trains an ELM on QA pairs, incorporating validation for early stopping.
func NewQAELMAgent(qas []QA, hiddenSize int, activation int, lambda float64, validationSplit float64, patience int) (*QAELMAgent, error) {
	// Shuffle and split the dataset into training and validation sets
	rand.Shuffle(len(qas), func(i, j int) {
		qas[i], qas[j] = qas[j], qas[i]
	})

	// Determine split index
	trainSize := int(float64(len(qas)) * (1 - validationSplit))
	trainData := qas[:trainSize]
	valData := qas[trainSize:]

	// Extract questions for training and validation
	trainQuestions := make([]string, len(trainData))
	for i, qa := range trainData {
		trainQuestions[i] = qa.Question
	}

	// Compute TF-IDF on all training questions to build vocabulary and idf
	tf := tfidf.NewTFIDF(trainQuestions)
	tf.CalculateScores()

	// Store idf values
	idf := make(map[string]float64, len(tf.InverseDocFreq))
	for term, val := range tf.InverseDocFreq {
		idf[term] = val
	}

	// Build sorted vocabulary from idf
	terms := make([]string, 0, len(idf))
	for term := range idf {
		terms = append(terms, term)
	}
	sort.Strings(terms)
	termToIndex := make(map[string]int, len(terms))
	for i, t := range terms {
		termToIndex[t] = i
	}

	// Extract unique answers
	answerToIndex := make(map[string]int)
	answers := make([]string, 0)
	for _, qa := range qas {
		if _, ok := answerToIndex[qa.Answer]; !ok {
			answerToIndex[qa.Answer] = len(answers)
			answers = append(answers, qa.Answer)
		}
	}

	// Prepare training matrices
	inputSize := len(terms)
	outputSize := len(answers)
	trainInputs := make([][]float64, len(trainData))
	trainTargets := make([][]float64, len(trainData))
	for i, qa := range trainData {
		// Vectorize question using same TF-IDF logic
		tfq := tfidf.NewTFIDF([]string{qa.Question})
		tfp := tfq.ProcessedWords
		vec := make([]float64, inputSize)
		total := float64(len(tfp))
		for _, w := range tfp {
			if idx, ok := termToIndex[w]; ok {
				if idfVal, ok2 := idf[w]; ok2 {
					vec[idx] += (1.0 / total) * idfVal
				}
			}
		}
		trainInputs[i] = vec
		// One-hot encode answer
		t := make([]float64, outputSize)
		ai := answerToIndex[qa.Answer]
		t[ai] = 1.0
		trainTargets[i] = t
	}

	// Prepare validation matrices similarly
	valInputs := make([][]float64, len(valData))
	valTargets := make([][]float64, len(valData))
	for i, qa := range valData {
		tfq := tfidf.NewTFIDF([]string{qa.Question})
		tfp := tfq.ProcessedWords
		vec := make([]float64, inputSize)
		total := float64(len(tfp))
		for _, w := range tfp {
			if idx, ok := termToIndex[w]; ok {
				if idfVal, ok2 := idf[w]; ok2 {
					vec[idx] += (1.0 / total) * idfVal
				}
			}
		}
		valInputs[i] = vec
		// One-hot encode answer
		t := make([]float64, outputSize)
		ai := answerToIndex[qa.Answer]
		t[ai] = 1.0
		valTargets[i] = t
	}

	// Initialize and train ELM model
	elmModel := elm.NewELM(inputSize, hiddenSize, outputSize, activation, lambda)
	elmModel.Train(trainInputs, trainTargets, valInputs, valTargets, patience)

	// 1) Initialize Beta to be hiddenSize×outputSize of zeros
	beta := mat.NewDense(hiddenSize, outputSize, nil)

	// 2) Initialize P = (λ I)^{-1} = (1/λ)·I
	p := mat.NewDense(hiddenSize, hiddenSize, nil)
	for i := 0; i < hiddenSize; i++ {
		p.Set(i, i, 1.0/lambda)
	}

	return &QAELMAgent{
		Model:         elmModel,
		terms:         terms,
		answers:       answers,
		termToIndex:   termToIndex,
		answerToIndex: answerToIndex,
		idf:           idf,
		P:             p,
		Beta:          beta,
		hiddenSize:    hiddenSize,
		lambda:        lambda,
		Corpus:        qas,
	}, nil
}

// Ask runs the trained ELM on a new question and returns the predicted answer.
func (agent *QAELMAgent) Ask(question string) (string, error) {
	// Process question tokens
	tfq := tfidf.NewTFIDF([]string{question})
	processed := tfq.ProcessedWords
	// Create input vector
	vec := make([]float64, len(agent.terms))
	total := float64(len(processed))
	for _, w := range processed {
		if idx, ok := agent.termToIndex[w]; ok {
			if idfVal, ok2 := agent.idf[w]; ok2 {
				vec[idx] += (1.0 / total) * idfVal
			}
		}
	}
	// Predict class scores
	out := agent.Model.Predict(vec)
	// Find top answer index
	maxIdx := 0
	maxVal := out[0]
	for i, v := range out {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	if maxIdx < len(agent.answers) {
		ans := agent.answers[maxIdx]
		if ans == "" {
			return "", errors.New("no answer found")
		}
		return ans, nil
	}
	return "", errors.New("prediction index out of range")
}

// Learn updates the ELM weights using the Woodbury identity.
func (a *QAELMAgent) Learn(newQA QA) error {
	a.mutex.Lock()
	defer a.mutex.Unlock()

	// 1) Append to the master corpus
	a.Corpus = append(a.Corpus, newQA)

	if err := a.Retrain(0.1 /* 10% val */, 5 /* patience */); err != nil {
		return err
	}

	// Check if Beta and P are initialized
	if a.Beta == nil {
		return fmt.Errorf("Beta matrix is not initialized")
	}
	if a.P == nil {
		return fmt.Errorf("P matrix is not initialized")
	}

	// Add new answer class if needed
	idx, exists := a.answerToIndex[newQA.Answer]
	if !exists {
		idx = len(a.answers)
		a.answerToIndex[newQA.Answer] = idx
		a.answers = append(a.answers, newQA.Answer)

		// Expand Beta and ELM output weights
		r, c := a.Beta.Dims() // Capture both dimensions
		newBeta := mat.NewDense(r, c+1, nil)
		newBeta.Slice(0, r, 0, c).(*mat.Dense).Copy(a.Beta)
		a.Beta = newBeta

		// Expand output weights in the model
		for i := range a.Model.OutputWeights {
			a.Model.OutputWeights[i] = append(a.Model.OutputWeights[i], 0)
		}
	}

	// Compute hidden activations
	x := a.vectorize(newQA.Question)
	h := a.Model.HiddenLayer(x)
	hVec := mat.NewVecDense(a.hiddenSize, h)

	// Make sure y is initialized correctly and has the right dimensions
	_, cols := a.Beta.Dims() // Capture dimensions properly
	// log.Printf("Beta dimensions: %v", a.Beta.Dims())
	log.Printf("Requesting target for index: %d (total classes: %d)", idx, cols)

	if cols <= 0 {
		return fmt.Errorf("Beta matrix has zero columns, cannot set target vector")
	}

	if idx >= cols {
		return fmt.Errorf("Index for target vector (%d) exceeds number of classes (%d)", idx, cols)
	}

	// Initialize the target vector with valid dimensions
	y := mat.NewVecDense(cols, nil)
	y.SetVec(idx, 1)

	// Woodbury update
	Ph := mat.NewVecDense(a.hiddenSize, nil)
	Ph.MulVec(a.P, hVec)
	denom := 1 + mat.Dot(hVec, Ph)

	// Delta = y - Beta^T h
	pred := mat.NewVecDense(cols, nil)
	pred.MulVec(a.Beta.T(), hVec)
	delta := mat.NewVecDense(cols, nil)
	delta.SubVec(y, pred)

	// Beta += (Ph outer delta) / denom
	outerBD := mat.NewDense(a.hiddenSize, cols, nil)
	outerBD.Outer(1.0/denom, Ph, delta)
	a.Beta.Add(a.Beta, outerBD)

	// Update output weights in the model
	for i := 0; i < a.hiddenSize; i++ {
		for j := 0; j < cols; j++ {
			a.Model.OutputWeights[i][j] = a.Beta.At(i, j)
		}
	}

	// Update P matrix
	outerPP := mat.NewDense(a.hiddenSize, a.hiddenSize, nil)
	outerPP.Outer(1.0/denom, Ph, Ph)
	a.P.Sub(a.P, outerPP)

	return nil
}

// vectorize converts text into TF-IDF feature vector.
func (a *QAELMAgent) vectorize(text string) []float64 {
	tf := tfidf.NewTFIDF([]string{text})
	tokens := tf.ProcessedWords
	vec := make([]float64, len(a.terms))
	total := float64(len(tokens))
	for _, w := range tokens {
		if idx, ok := a.termToIndex[w]; ok {
			vec[idx] += a.idf[w] / total
		}
	}
	return vec
}

// Updated ParseMarkdownQA function
func ParseMarkdownQA(corpus []string) []QA {
	var qas []QA
	// Regex to match question headings like '## Q: question text'
	qRegex := regexp.MustCompile(`^##\s*Q:\s*(.+)$`)
	heading := regexp.MustCompile(`^##`)
	splitLines := regexp.MustCompile("\r?\n")

	for _, doc := range corpus {
		lines := splitLines.Split(doc, -1)
		for i, line := range lines {
			if m := qRegex.FindStringSubmatch(line); m != nil {
				question := strings.TrimSpace(m[1])
				var ansBuilder strings.Builder

				// Gather answer lines until the next heading
				for j := i + 1; j < len(lines) && !heading.MatchString(lines[j]); j++ {
					l := strings.TrimSpace(lines[j])
					if l != "" {
						ansBuilder.WriteString(l)
						ansBuilder.WriteByte(' ')
					}
				}

				answer := strings.TrimSpace(ansBuilder.String())
				if answer == "" {
					// Skip any QA with no real answer
					continue
				}
				qas = append(qas, QA{Question: question, Answer: answer})
			}
		}

	}
	return qas
}

// Function to generate questions and answers from text using NLP and save to a file
func generateQuestionsAnswers(text string) []QA {
	var qas []QA

	// Create a new document from the text
	doc, err := prose.NewDocument(text)
	if err != nil {
		return qas // Return empty if there's an error
	}

	// Extract sentences for potential questions
	sentences := doc.Sentences()

	// Loop through sentences, looking for patterns
	for i := 0; i < len(sentences); i++ {
		sentence := sentences[i].Text
		trimmed := strings.TrimSpace(sentence)

		// Check for descriptive sentences to generate questions
		if strings.Contains(trimmed, "is") || strings.Contains(trimmed, "are") || strings.Contains(trimmed, "provides") {
			// Formulate a question from a statement
			question := "What " + strings.ToLower(trimmed[:1]) + trimmed[1:] // Make it a question
			question = strings.ReplaceAll(question, " the ", " ")            // Remove if it's "the"
			question = strings.ReplaceAll(question, " is ", " ? ")           // Convert to question format
			answer := trimmed                                                // Use the same sentence as the answer

			// Add the QA pair
			if answer != "" {
				qas = append(qas, QA{Question: question, Answer: answer})
			}
		}

		// Creating questions based on code examples
		if strings.Contains(trimmed, "func") {
			// Generate questions from function definitions
			question := "How to define a function in Go?"
			answer := trimmed

			// Enrich the answer with context
			if i+1 < len(sentences) {
				answer += " Example: " + strings.TrimSpace(sentences[i+1].Text)
			}

			qas = append(qas, QA{Question: question, Answer: answer})
		}
	}

	// Save the generated QAs to a markdown file
	err = saveQAsToFile(qas, "corpus/generated.md")
	if err != nil {
		// Handle file saving error (optional)
		return nil
	}

	return qas
}

// Function to save QAs to a markdown file
func saveQAsToFile(qas []QA, filepath string) error {
	// Create or open the file
	file, err := os.Create(filepath)
	if err != nil {
		return err // Return error if the file cannot be created/opened
	}
	defer file.Close() // Ensure the file is closed after writing

	// Write the QAs to the file
	for _, qa := range qas {
		_, err = file.WriteString("## " + qa.Question + "\n\n")
		if err != nil {
			return err // Handle potential writing error
		}
		_, err = file.WriteString(qa.Answer + "\n\n---\n\n") // Markdown separation
		if err != nil {
			return err
		}
	}

	return nil
}

// Retrain rebuilds the ELM from the full QA corpus using the same parameters.
func (a *QAELMAgent) Retrain(validationSplit float64, patience int) error {
	// Re-run the training pipeline on the entire corpus
	newAgent, err := NewQAELMAgent(a.Corpus, a.hiddenSize, a.Model.Activation, a.lambda, validationSplit, patience)
	if err != nil {
		return fmt.Errorf("retraining failed: %w", err)
	}
	// Swap internal state
	a.Model = newAgent.Model
	a.terms = newAgent.terms
	a.answers = newAgent.answers
	a.termToIndex = newAgent.termToIndex
	a.answerToIndex = newAgent.answerToIndex
	a.idf = newAgent.idf
	a.P = newAgent.P
	a.Beta = newAgent.Beta
	return nil
}
