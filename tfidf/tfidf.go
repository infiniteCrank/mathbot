package tfidf

import (
	"math"
	"regexp"
	"sort"
	"strings"
)

// TFIDF struct holds the term frequency and inverse document frequency.
type TFIDF struct {
	TermFrequency  map[string]float64 // Frequencies of terms (and n-grams) in the corpus
	InverseDocFreq map[string]float64 // Inverse document frequencies for terms
	WordsInDoc     []string           // Raw tokens extracted from the corpus
	ProcessedWords []string           // Processed tokens (filtered, stemmed, lemmatized, plus n-grams)
	Scores         map[string]float64 // Calculated TF-IDF scores from TF and IDF
	TopKeyWords    map[string]float64 // The top X keywords in the corpus
	Corpus         string             // The plain text corpus (concatenated docs)
}

// NewTFIDF creates a new TFIDF instance based on the provided corpus of documents.
func NewTFIDF(corpus []string) *TFIDF {
	tf := make(map[string]float64)  // term frequencies
	idf := make(map[string]float64) // inverse document frequencies
	var wholeCorpus string
	var wordsFinal []string
	re := regexp.MustCompile("[^a-zA-Z0-9]+")

	// Process each document in the corpus.
	for _, doc := range corpus {
		wholeCorpus += doc
		words := strings.Fields(doc)
		for _, word := range words {
			// Replace punctuation with space.
			cleanedWord := re.ReplaceAllString(word, " ")
			// Split again in case punctuation removal resulted in multiple tokens.
			tokens := strings.Fields(cleanedWord)
			wordsFinal = append(wordsFinal, tokens...)
		}
	}

	// Process words including filtering, stemming, lemmatization, and n-gram generation.
	processedWords := processWords(wordsFinal)

	// Compute term frequencies.
	for _, token := range processedWords {
		tf[token]++
	}

	// Compute inverse document frequencies.
	for term := range tf {
		idf[term] = math.Log(float64(len(corpus)) / (1 + float64(countDocumentsContainingTerm(corpus, term))))
	}

	return &TFIDF{
		TermFrequency:  tf,
		InverseDocFreq: idf,
		WordsInDoc:     wordsFinal,
		ProcessedWords: processedWords,
		Corpus:         wholeCorpus,
	}
}

// countDocumentsContainingTerm counts how many documents in the corpus contain the given term.
func countDocumentsContainingTerm(corpus []string, term string) int {
	count := 0
	for _, doc := range corpus {
		if strings.Contains(doc, term) {
			count++
		}
	}
	return count
}

// CalculateScores computes the TF-IDF score for each token in the processed words.
// A TF-IDF score measures how important a word is to a specific document within a
// collection of documents, taking into account how often the word appears in that
// document (term frequency) and how rare it is across all documents in the collection
// (inverse document frequency) - essentially giving more weight to words that are frequent
// within a specific document but uncommon across the whole set of documents.
func (tfidf *TFIDF) CalculateScores() map[string]float64 {
	scores := make(map[string]float64)
	totalWords := float64(len(tfidf.ProcessedWords))
	for _, token := range tfidf.ProcessedWords {
		if freq, exists := tfidf.TermFrequency[token]; exists {
			scores[token] = (freq / totalWords) * tfidf.InverseDocFreq[token]
		}
	}
	tfidf.Scores = scores
	return scores
}

// processWords applies stop word removal, advanced stemming, lemmatization,
// and generates bi-grams and tri-grams.
func processWords(words []string) []string {
	// Remove stop words and apply advanced stemming.
	filtered := removeStopWordsAndAdvancedStem(words)

	// Apply lemmatization.
	for i, word := range filtered {
		filtered[i] = lemmatize(word)
	}

	// Generate bi-grams and tri-grams from the filtered tokens.
	bigrams := generateNGrams(filtered, 2)
	trigrams := generateNGrams(filtered, 3)

	// Combine unigrams, bi-grams, and tri-grams.
	allTokens := append(filtered, bigrams...)
	allTokens = append(allTokens, trigrams...)
	return allTokens
}

// generateNGrams creates n-grams from a slice of tokens.
func generateNGrams(tokens []string, n int) []string {
	var ngrams []string
	if len(tokens) < n {
		return ngrams
	}
	for i := 0; i <= len(tokens)-n; i++ {
		ngram := strings.Join(tokens[i:i+n], " ")
		ngrams = append(ngrams, ngram)
	}
	return ngrams
}

// lemmatize converts a word to its base form using custom rules.
func lemmatize(word string) string {
	lemmatizationRules := map[string]string{
		"execute":    "execute",
		"running":    "run",
		"returns":    "return",
		"defined":    "define",
		"compiles":   "compile",
		"calls":      "call",
		"creating":   "create",
		"invoke":     "invoke",
		"declares":   "declare",
		"references": "reference",
		"implements": "implement",
		"utilizes":   "utilize",
		"tests":      "test",
		"loops":      "loop",
		"deletes":    "delete",
		"functions":  "function",
	}
	if base, exists := lemmatizationRules[word]; exists {
		return base
	}
	// Basic removal of common suffixes as fallback.
	if strings.HasSuffix(word, "ing") {
		return word[:len(word)-len("ing")]
	}
	if strings.HasSuffix(word, "ed") {
		return word[:len(word)-len("ed")]
	}
	return word
}

// removeStopWordsAndAdvancedStem removes common stop words and applies advanced stemming.
func removeStopWordsAndAdvancedStem(words []string) []string {
	stopWords := map[string]struct{}{
		"a": {}, "and": {}, "the": {}, "is": {}, "to": {},
		"of": {}, "in": {}, "it": {}, "that": {}, "you": {},
		"this": {}, "for": {}, "on": {}, "are": {}, "with": {},
		"as": {}, "be": {}, "by": {}, "at": {}, "from": {},
		"or": {}, "an": {}, "but": {}, "not": {}, "we": {},
	}
	var filtered []string
	for _, word := range words {
		if _, exists := stopWords[word]; !exists {
			stemmedWord := advancedStem(word)
			filtered = append(filtered, stemmedWord)
		}
	}
	return filtered
}

// advancedStem applies simple suffix removal based on common suffixes.
func advancedStem(word string) string {
	suffixes := []string{"es", "ed", "ing", "s", "ly", "ment", "ness", "ity", "ism", "er"}
	programmingKeywords := map[string]struct{}{
		"func": {}, "package": {}, "import": {}, "interface": {}, "go": {},
		"goroutine": {}, "channel": {}, "select": {}, "struct": {},
		"map": {}, "slice": {}, "var": {}, "const": {}, "type": {},
		"defer": {}, "fallthrough": {},
	}
	if _, isKeyword := programmingKeywords[word]; isKeyword {
		return word
	}
	for _, suffix := range suffixes {
		if strings.HasSuffix(word, suffix) {
			// Special handling for suffix "es" preceded by "i"
			if suffix == "es" && len(word) > 2 && word[len(word)-3] == 'i' {
				return word[:len(word)-2]
			}
			return word[:len(word)-len(suffix)]
		}
	}
	return word
}

// ExtractKeywords returns the top N keywords from the computed TF-IDF scores.
func (tfidf *TFIDF) ExtractKeywords(topN int) map[string]float64 {
	type kv struct {
		Key   string
		Value float64
	}
	var sortedTerms []kv
	for k, v := range tfidf.Scores {
		sortedTerms = append(sortedTerms, kv{k, v})
	}
	sort.Slice(sortedTerms, func(i, j int) bool {
		return sortedTerms[i].Value > sortedTerms[j].Value
	})
	topKeywords := make(map[string]float64)
	for i := 0; i < topN && i < len(sortedTerms); i++ {
		topKeywords[sortedTerms[i].Key] = sortedTerms[i].Value
	}
	return topKeywords
}

// Add a method to return the index of processed words in TF-IDF
func (tfidf *TFIDF) ProcessedWordsIndex(word string) int {
	for i, w := range tfidf.ProcessedWords {
		if w == word {
			return i
		}
	}
	return -1 // Not found
}
