package fileLoader

import (
	"log"
	"os"
	"path/filepath"
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
