package markdownenv

import (
	"math/rand"
	"time"

	"github.com/infiniteCrank/mathbot/tfidf"
)

// MarkdownEnv implements the agent.Environment interface.
// It now accepts a slice of markdown documents for training.
type MarkdownEnv struct {
	Corpus           []string     // The collection of markdown documents.
	TFIDF            *tfidf.TFIDF // TFIDF instance built from the corpus.
	TopKeywords      []string     // Top keywords extracted from the corpus.
	StateVector      []float64    // State vector representing the TFIDF scores for the top keywords.
	ActionSpaceIndex int          // Number of possible actions (keywords).
	TargetIndex      int          // Randomly chosen target keyword for the current episode.
	Done             bool         // Indicates the episode is finished.
}

// NewMarkdownEnvFromCorpus creates an environment from multiple markdown documents.
// topN is the number of top keywords to extract.
func NewMarkdownEnvFromCorpus(corpus []string, topN int) *MarkdownEnv {
	// Create the TFIDF instance from the corpus.
	tfidfInstance := tfidf.NewTFIDF(corpus)
	tfidfInstance.CalculateScores()
	top := tfidfInstance.ExtractKeywords(topN)

	// For a consistent ordering, we may want to collect the keys in a defined order.
	var sortedKeywords []string
	for kw := range top {
		sortedKeywords = append(sortedKeywords, kw)
	}
	// Optionally sort (alphabetically or by score) if desired:
	// sort.Strings(sortedKeywords)

	// Build the state vector in the same order.
	stateVec := make([]float64, len(sortedKeywords))
	for i, kw := range sortedKeywords {
		stateVec[i] = top[kw]
	}

	return &MarkdownEnv{
		Corpus:           corpus,
		TFIDF:            tfidfInstance,
		TopKeywords:      sortedKeywords,
		StateVector:      stateVec,
		ActionSpaceIndex: len(sortedKeywords),
		Done:             false,
	}
}

// Reset resets the environment at the start of each episode.
// It randomly selects one of the keywords as the target.
func (env *MarkdownEnv) Reset() []float64 {
	env.Done = false
	rand.Seed(time.Now().UnixNano())
	env.TargetIndex = rand.Intn(env.ActionSpaceIndex)
	return env.StateVector
}

// Step takes an action (an index) and returns the state, reward, and done flag.
// The reward is 1 if the chosen action matches the target index.
func (env *MarkdownEnv) Step(action int) (nextState []float64, reward float64, done bool) {
	if action == env.TargetIndex {
		reward = 1.0
	} else {
		reward = 0.0
	}
	env.Done = true // One-step episode.
	return env.StateVector, reward, env.Done
}

// ActionSpace returns the number of keywords.
func (env *MarkdownEnv) ActionSpace() int {
	return env.ActionSpaceIndex
}

// StateDimensions returns the dimension of the state vector.
func (env *MarkdownEnv) StateDimensions() int {
	return len(env.StateVector)
}
