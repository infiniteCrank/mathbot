package tfidf

import (
	"math"
	"regexp"
	"sort"
	"strings"
)

// TFIDF struct holds term frequencies, inverse document frequencies,
// and a fixed vocabulary/index for transforming new documents.
type TFIDF struct {
	TermFrequency  map[string]float64 // Frequencies of terms in the training corpus
	InverseDocFreq map[string]float64 // IDF values per term
	ProcessedWords []string           // All tokens (unigrams, n-grams) from corpus
	Corpus         string             // Concatenated corpus text

	// Vocabulary and index map for feature vectors
	Vocabulary []string       // Sorted unique terms
	Index      map[string]int // term -> index in Vocabulary

	// Scores map holds the last computed TF-IDF scores for analysis
	Scores map[string]float64
}

// NewTFIDF computes TF and IDF over the provided corpus and builds the vocab.
func NewTFIDF(corpus []string) *TFIDF {
	re := regexp.MustCompile(`[^a-zA-Z0-9]+`)
	var wordsFinal []string
	var wholeCorpus strings.Builder
	for _, doc := range corpus {
		wholeCorpus.WriteString(doc)
		for _, tok := range strings.Fields(doc) {
			clean := re.ReplaceAllString(tok, " ")
			wordsFinal = append(wordsFinal, strings.Fields(clean)...)
		}
	}

	processed := processWords(wordsFinal)

	tf := make(map[string]float64)
	for _, w := range processed {
		tf[w]++
	}

	idf := make(map[string]float64)
	for term := range tf {
		idf[term] = math.Log(float64(len(corpus)) / (1 + float64(countDocumentsContainingTerm(corpus, term))))
	}

	t := &TFIDF{
		TermFrequency:  tf,
		InverseDocFreq: idf,
		ProcessedWords: processed,
		Corpus:         wholeCorpus.String(),
		Scores:         make(map[string]float64),
	}
	t.BuildVocab()
	return t
}

// BuildVocab constructs the sorted Vocabulary and Index map.
func (t *TFIDF) BuildVocab() {
	unique := make(map[string]struct{}, len(t.ProcessedWords))
	for _, w := range t.ProcessedWords {
		unique[w] = struct{}{}
	}
	t.Vocabulary = make([]string, 0, len(unique))
	for w := range unique {
		t.Vocabulary = append(t.Vocabulary, w)
	}
	sort.Strings(t.Vocabulary)
	t.Index = make(map[string]int, len(t.Vocabulary))
	for i, w := range t.Vocabulary {
		t.Index[w] = i
	}
}

// CalculateScores computes TF-IDF scores for each term in the training corpus.
// It populates t.Scores and returns the map of term -> score.
func (t *TFIDF) CalculateScores() map[string]float64 {
	scores := make(map[string]float64, len(t.TermFrequency))
	total := float64(len(t.ProcessedWords))
	for term, freq := range t.TermFrequency {
		if idf, ok := t.InverseDocFreq[term]; ok {
			scores[term] = (freq / total) * idf
		}
	}
	t.Scores = scores
	return scores
}

// ExtractKeywords returns the top N terms by TF-IDF score.
// Make sure to call CalculateScores() first to populate t.Scores.
func (t *TFIDF) ExtractKeywords(topN int) map[string]float64 {
	terms := make([]struct {
		Key   string
		Value float64
	}, 0, len(t.Scores))
	for k, v := range t.Scores {
		terms = append(terms, struct {
			Key   string
			Value float64
		}{k, v})
	}
	sort.Slice(terms, func(i, j int) bool {
		return terms[i].Value > terms[j].Value
	})
	top := make(map[string]float64, topN)
	for i := 0; i < topN && i < len(terms); i++ {
		top[terms[i].Key] = terms[i].Value
	}
	return top
}

// Transform computes the TF-IDF vector for a new document (string).
func (t *TFIDF) Transform(doc string) []float64 {
	re := regexp.MustCompile(`[^a-zA-Z0-9]+`)
	var raw []string
	for _, tok := range strings.Fields(doc) {
		clean := re.ReplaceAllString(tok, " ")
		raw = append(raw, strings.Fields(clean)...)
	}
	processed := processWords(raw)

	tfDoc := make(map[string]float64)
	for _, w := range processed {
		tfDoc[w]++
	}
	total := float64(len(processed))

	vec := make([]float64, len(t.Vocabulary))
	for term, cnt := range tfDoc {
		if idx, ok := t.Index[term]; ok {
			vec[idx] = (cnt / total) * t.InverseDocFreq[term]
		}
	}
	return vec
}

// TransformBatch applies Transform to each document.
func (t *TFIDF) TransformBatch(docs []string) [][]float64 {
	out := make([][]float64, len(docs))
	for i, d := range docs {
		out[i] = t.Transform(d)
	}
	return out
}

// countDocumentsContainingTerm counts docs in which term appears.
func countDocumentsContainingTerm(corpus []string, term string) int {
	cnt := 0
	for _, doc := range corpus {
		if strings.Contains(doc, term) {
			cnt++
		}
	}
	return cnt
}

// --- existing processing helpers ---
func processWords(words []string) []string {
	filtered := removeStopWordsAndAdvancedStem(words)
	for i, w := range filtered {
		filtered[i] = lemmatize(w)
	}
	bigrams := generateNGrams(filtered, 2)
	trigrams := generateNGrams(filtered, 3)
	all := append(filtered, bigrams...)
	all = append(all, trigrams...)
	return all
}

func generateNGrams(tokens []string, n int) []string {
	var ngrams []string
	if len(tokens) < n {
		return ngrams
	}
	for i := 0; i <= len(tokens)-n; i++ {
		ngrams = append(ngrams, strings.Join(tokens[i:i+n], " "))
	}
	return ngrams
}

func lemmatize(word string) string {
	rules := map[string]string{
		"running": "run", "returns": "return", "defined": "define",
		// ...other rules...
	}
	if base, ok := rules[word]; ok {
		return base
	}
	if strings.HasSuffix(word, "ing") {
		return word[:len(word)-3]
	}
	if strings.HasSuffix(word, "ed") {
		return word[:len(word)-2]
	}
	return word
}

func removeStopWordsAndAdvancedStem(words []string) []string {
	stops := map[string]struct{}{ /* same stop-words set as before */ }
	var out []string
	for _, w := range words {
		if _, bad := stops[w]; !bad {
			out = append(out, advancedStem(w))
		}
	}
	return out
}

func advancedStem(word string) string {
	suffixes := []string{"es", "ed", "ing", "s", "ly", "ment", "ness", "ity", "ism", "er"}
	keywords := map[string]struct{}{ /* programming keywords */ }
	if _, ok := keywords[word]; ok {
		return word
	}
	for _, suf := range suffixes {
		if strings.HasSuffix(word, suf) {
			if suf == "es" && len(word) > 2 && word[len(word)-3] == 'i' {
				return word[:len(word)-2]
			}
			return word[:len(word)-len(suf)]
		}
	}
	return word
}
