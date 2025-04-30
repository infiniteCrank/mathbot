// Package retrieval provides a simple TF-IDF based document retriever
// that can chunk arbitrary markdown/text into passages and answer queries by
// returning the most similar passage.
package retrieval

import (
	"math"
	"regexp"
	"strings"

	"github.com/infiniteCrank/mathbot/tfidf"
)

// Retriever holds a TF-IDF model over a set of docs and their vectors.
type Retriever struct {
	TfIdf      *tfidf.TFIDF
	Docs       []string    // raw document chunks
	DocVectors [][]float64 // TF-IDF vectors for each chunk
}

// NewRetriever builds a retriever over the given chunks.
func NewRetriever(chunks []string) *Retriever {
	// Build TF-IDF on all chunks
	t := tfidf.NewTFIDF(chunks)
	// Compute all document vectors
	vecs := t.TransformBatch(chunks)
	return &Retriever{
		TfIdf:      t,
		Docs:       chunks,
		DocVectors: vecs,
	}
}

// CosineSimilarity computes cosine similarity between two vectors.
func CosineSimilarity(a, b []float64) float64 {
	var dot, magA, magB float64
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		magA += a[i] * a[i]
		magB += b[i] * b[i]
	}
	if magA == 0 || magB == 0 {
		return 0
	}
	return dot / (math.Sqrt(magA) * math.Sqrt(magB))
}

// Query returns the most similar chunk and its score for the given query string.
func (r *Retriever) Query(q string) (bestDoc string, bestScore float64) {
	qv := r.TfIdf.Transform(q)
	bestScore = -1
	for i, dv := range r.DocVectors {
		s := CosineSimilarity(qv, dv)
		if s > bestScore {
			bestScore = s
			bestDoc = r.Docs[i]
		}
	}
	return
}

// ChunkMarkdown splits markdown text into chunks at headings of the given level,
// e.g. level=2 splits at lines starting with "## ".
func ChunkMarkdown(md string, level int) []string {
	re := regexp.MustCompile(`(?m)^#{` + regexp.QuoteMeta(string('0'+level)) + `}\s+`)
	lines := strings.Split(md, "\n")
	var chunks []string
	var buf []string
	for _, line := range lines {
		if level > 0 && re.MatchString(line) {
			// start of new chunk
			if len(buf) > 0 {
				chunks = append(chunks, strings.Join(buf, "\n"))
			}
			buf = []string{line}
		} else {
			buf = append(buf, line)
		}
	}
	if len(buf) > 0 {
		chunks = append(chunks, strings.Join(buf, "\n"))
	}
	return chunks
}
