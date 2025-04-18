package agent

import (
	"errors"
	"regexp"
	"sort"

	"github.com/infiniteCrank/mathbot/elm"
	"github.com/infiniteCrank/mathbot/tfidf"
)

// QA holds a question-answer pair extracted from markdown.
type QA struct {
	Question string
	Answer   string
}

// QAELMAgent wraps an ELM for QA tasks.
type QAELMAgent struct {
	model         *elm.ELM
	terms         []string           // sorted vocabulary terms
	answers       []string           // list of unique answers
	termToIndex   map[string]int     // maps term to vector index
	answerToIndex map[string]int     // maps answer to class index
	idf           map[string]float64 // inverse document frequency from training corpus
}

// NewQAELMAgent builds and trains an ELM on QA pairs.
func NewQAELMAgent(qas []QA, hiddenSize int, activation int, lambda float64) (*QAELMAgent, error) {
	// Extract questions
	questions := make([]string, len(qas))
	for i, qa := range qas {
		questions[i] = qa.Question
	}
	// Compute TF-IDF on all questions to build vocabulary and idf
	tf := tfidf.NewTFIDF(questions)
	tf.CalculateScores()
	// Store idf values
	idf := make(map[string]float64, len(tf.InverseDocFreq))
	for term, val := range tf.InverseDocFreq {
		idf[term] = val
	}
	// Build sorted vocabulary
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
	trainInputs := make([][]float64, len(qas))
	trainTargets := make([][]float64, len(qas))
	for i, qa := range qas {
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
	// Initialize and train ELM
	elmModel := elm.NewELM(inputSize, hiddenSize, outputSize, activation, lambda)
	elmModel.Train(trainInputs, trainTargets)

	return &QAELMAgent{
		model:         elmModel,
		terms:         terms,
		answers:       answers,
		termToIndex:   termToIndex,
		answerToIndex: answerToIndex,
		idf:           idf,
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
	out := agent.model.Predict(vec)
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
		return agent.answers[maxIdx], nil
	}
	return "", errors.New("prediction index out of range")
}

// ParseMarkdownQA extracts QA pairs from markdown corpus.
func ParseMarkdownQA(corpus []string) []QA {
	var qas []QA
	// regex to match question headings like '## Q: question text'
	r := regexp.MustCompile(`^##\s*Q:\s*(.+)$`)
	splitLines := regexp.MustCompile("\r?\n")
	for _, doc := range corpus {
		lines := splitLines.Split(doc, -1)
		for i, line := range lines {
			if m := r.FindStringSubmatch(line); m != nil {
				q := m[1]
				// Collect answer lines until next heading
				ans := ""
				for j := i + 1; j < len(lines) && !regexp.MustCompile(`^##`).MatchString(lines[j]); j++ {
					if lines[j] != "" {
						ans += lines[j] + " "
					}
				}
				qas = append(qas, QA{Question: q, Answer: ans})
			}
		}
	}
	return qas
}
