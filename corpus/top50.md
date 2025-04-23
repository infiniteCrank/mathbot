## Q: What is Go (Golang)?

Go is an open-source programming language developed by Google. It is statically typed, compiled, and known for its simplicity, efficiency, and excellent support for concurrency.

## Q: What are the key features of Go?

Key features include simplicity, concurrency support via goroutines, fast compilation, strong standard library, garbage collection, and statically linked binaries.

## Q: How do you declare a variable in Go?

Use the var keyword: var x int = 10 or shorthand: x := 10.

## Q: What is a goroutine?

A goroutine is a lightweight thread managed by the Go runtime. Use go keyword: go myFunction().

## Q: How do channels work in Go?

Channels are used for communication between goroutines: ch := make(chan int), ch <- 1, val := <- ch.

## Q: What is a slice in Go?

A slice is a dynamically-sized, flexible view into arrays. Example: s := []int{1, 2, 3}.

## Q: How does Go handle memory management?

Go uses garbage collection to automatically manage memory.

## Q: What is the difference between an array and a slice?

Arrays have fixed size; slices are dynamic and more commonly used.

## Q: How do you handle errors in Go?

By returning error values: val, err := someFunc(); if err != nil { // handle }

## Q: What is a struct in Go?

A struct is a composite type used to group fields: type Person struct { Name string; Age int }

## Q: How does Go support object-oriented programming?

Go uses structs and interfaces instead of classes. It supports composition over inheritance.

## Q: What is an interface in Go?

An interface defines a set of method signatures. A type implements an interface by defining its methods.

## Q: What are Go modules?

Modules are the dependency management system introduced in Go 1.11. Use go mod init to create a module.

## Q: How do you write a function in Go?

func add(a int, b int) int { return a + b }

## Q: Can you return multiple values from a function?

Yes. func swap(a, b string) (string, string) { return b, a }

## Q: What is a pointer in Go?

A pointer holds the address of a variable. Use * and & to dereference and reference.

## Q: How do you create a map in Go?

Use make: m := make(map[string]int)

## Q: How do you delete a key from a map?

Use delete(m, key).

## Q: What is the blank identifier _ used for?

To ignore values: _, err := someFunc()

## Q: What is defer in Go?

defer schedules a function to run after the surrounding function returns.

## Q: How do you handle panics in Go?

Use recover in a deferred function to catch a panic.

## Q: What is the init function in Go?

It runs before the main function and is used for setup.

## Q: What are Go's visibility rules?

Identifiers starting with a capital letter are exported (public). Lowercase means unexported (private).

## Q: How do you run a Go program?

Use go run file.go.

## Q: How do you build a Go binary?

Use go build.

## Q: How do you test code in Go?

Use the testing package. Tests go in files ending with _test.go.

## Q: How do you write a test function?

func TestXxx(t *testing.T) { }

## Q: How do you benchmark in Go?

Use func BenchmarkXxx(b *testing.B).

## Q: What are Go routines and how are they different from threads?

Goroutines are managed by the Go runtime, more lightweight than threads.

## Q: How do you synchronize goroutines?

Use channels or the sync package (e.g., sync.WaitGroup, sync.Mutex).

## Q: What is the difference between go run and go build?

go run compiles and runs the program. go build only compiles.

## Q: What is the Go workspace?

It's the directory structure for Go projects using GOPATH (legacy). Modules are now preferred.

## Q: What is context in Go?

The context package allows propagation of deadlines, cancellation signals, and values.

## Q: How do you format Go code?

Use gofmt or go fmt.

## Q: What is a closure in Go?

A closure is a function that captures variables from its surrounding scope.

## Q: How do you create a constant in Go?

Use const: const Pi = 3.14

## Q: What is embedding in Go?

Embedding allows a struct to include another struct and its fields/methods.

## Q: What is type assertion?

Used to extract the dynamic type from an interface: val, ok := i.(T)

## Q: What is a type switch?

A switch for interface types: switch v := i.(type) { ... }

## Q: What is the zero value in Go?

Default value for a type: 0 for ints, "" for strings, nil for pointers/slices/maps.

## Q: Can Go support generics?

Yes, generics were introduced in Go 1.18 using type parameters.

## Q: How do you handle JSON in Go?

Use the encoding/json package to marshal and unmarshal.

## Q: What is the difference between new and make?

new allocates memory; make initializes slices, maps, or channels.

## Q: How do you do dependency injection in Go?

Manually pass dependencies via constructors. No built-in DI framework.

## Q: What is race condition and how to detect it?

A race condition happens with concurrent writes. Use go run -race to detect.

## Q: How do you work with files in Go?

Use os and io/ioutil or bufio packages.

## Q: How do you create custom packages?

Create a directory with a .go file and use package packagename.

## Q: How do you document code in Go?

Use comments above declarations. godoc extracts documentation.

## Q: What are common Go tools?

go run, go build, go fmt, go test, go mod, golint, go vet.

## Q: How do you handle concurrency safely?

Use channels and synchronization primitives like mutexes and atomic operations.