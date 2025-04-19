package fileLoader

import (
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// LoadMarkdownFiles scans the given directory for .md files,
// reads each file, and returns a slice of strings (one per document).
func LoadMarkdownFiles(directory string) ([]string, error) {
	var docs []string

	// Find all markdown files in the directory.
	paths, err := filepath.Glob(filepath.Join(directory, "*.md"))
	if err != nil {
		return nil, err
	}

	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			log.Printf("error reading file %s: %v", path, err)
			continue
		}
		docs = append(docs, string(data))
	}
	return docs, nil
}

// processMarkdownContent processes a single Markdown document and extracts questions and answers
func ProcessMarkdownContent(content string, questionsAndAnswers map[string]string) {
	// Split content into lines
	lines := strings.Split(content, "\n")
	var currentSection string
	var accumulatedContent strings.Builder

	// Regex for section headings
	headingRegex := regexp.MustCompile(`^##+\s+(.*)`)

	// Read the file line by line
	for _, line := range lines {
		// Check if the line is a heading
		if matches := headingRegex.FindStringSubmatch(line); matches != nil {
			if accumulatedContent.Len() > 0 {
				// Generate a question from the previous section and its content
				questionsAndAnswers[currentSection] = accumulatedContent.String()
				accumulatedContent.Reset() // Clear the builder for next content
			}
			currentSection = matches[1] // Set new section title as question
		} else {
			accumulatedContent.WriteString(line + "\n") // Accumulate content for the current section
		}
	}

	// Handle the last section
	if currentSection != "" && accumulatedContent.Len() > 0 {
		questionsAndAnswers[currentSection] = accumulatedContent.String()
	}
}
