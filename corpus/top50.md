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

## Q: What is Go (Golang)?
Go is an open-source programming language developed by Google. It is statically typed, compiled, and known for its simplicity, efficiency, and excellent support for concurrency.

## Q: What are the key features of Go?
Key features include simplicity, concurrency support via goroutines, fast compilation, strong standard library, garbage collection, and statically linked binaries.

## Q: How do you declare a variable in Go?
Use the `var` keyword: `var x int = 10` or shorthand: `x := 10`.

## Q: What is a goroutine?
A goroutine is a lightweight thread managed by the Go runtime. Use `go` keyword: `go myFunction()`.

## Q: How do channels work in Go?
Channels are used for communication between goroutines: `ch := make(chan int)`, `ch <- 1`, `val := <- ch`.

## Q: What is a slice in Go?
A slice is a dynamically-sized, flexible view into arrays. Example: `s := []int{1, 2, 3}`.

## Q: How does Go handle memory management?
Go uses garbage collection to automatically manage memory.

## Q: What is the difference between an array and a slice?
Arrays have fixed size; slices are dynamic and more commonly used.

## Q: How do you handle errors in Go?
By returning error values: `val, err := someFunc(); if err != nil { // handle }`

## Q: What is a struct in Go?
A struct is a composite type used to group fields: `type Person struct { Name string; Age int }`

## Q: How does Go support object-oriented programming?
Go uses structs and interfaces instead of classes. It supports composition over inheritance.

## Q: What is an interface in Go?
An interface defines a set of method signatures. A type implements an interface by defining its methods.

## Q: What are Go modules?
Modules are the dependency management system introduced in Go 1.11. Use `go mod init` to create a module.

## Q: How do you write a function in Go?
`func add(a int, b int) int { return a + b }`

## Q: Can you return multiple values from a function?
Yes. `func swap(a, b string) (string, string) { return b, a }`

## Q: What is a pointer in Go?
A pointer holds the address of a variable. Use `*` and `&` to dereference and reference.

## Q: How do you create a map in Go?
Use `make`: `m := make(map[string]int)`

## Q: How do you delete a key from a map?
Use `delete(m, key)`.

## Q: What is the blank identifier `_` used for?
To ignore values: `_, err := someFunc()`

## Q: What is defer in Go?
`defer` schedules a function to run after the surrounding function returns.

## Q: How do you handle panics in Go?
Use `recover` in a deferred function to catch a panic.

## Q: What is the `init` function in Go?
It runs before the main function and is used for setup.

## Q: What are Go's visibility rules?
Identifiers starting with a capital letter are exported (public). Lowercase means unexported (private).

## Q: How do you run a Go program?
Use `go run file.go`.

## Q: How do you build a Go binary?
Use `go build`.

## Q: How do you test code in Go?
Use the testing package. Tests go in files ending with `_test.go`.

## Q: How do you write a test function?
`func TestXxx(t *testing.T) { }`

## Q: How do you benchmark in Go?
Use `func BenchmarkXxx(b *testing.B)`.

## Q: What are Go routines and how are they different from threads?
Goroutines are managed by the Go runtime, more lightweight than threads.

## Q: How do you synchronize goroutines?
Use channels or the `sync` package (e.g., `sync.WaitGroup`, `sync.Mutex`).

## Q: What is the difference between `go run` and `go build`?
`go run` compiles and runs the program. `go build` only compiles.

## Q: What is the Go workspace?
It's the directory structure for Go projects using GOPATH (legacy). Modules are now preferred.

## Q: What is context in Go?
The `context` package allows propagation of deadlines, cancellation signals, and values.

## Q: How do you format Go code?
Use `gofmt` or `go fmt`.

## Q: What is a closure in Go?
A closure is a function that captures variables from its surrounding scope.

## Q: How do you create a constant in Go?
Use `const`: `const Pi = 3.14`

## Q: What is embedding in Go?
Embedding allows a struct to include another struct and its fields/methods.

## Q: What is type assertion?
Used to extract the dynamic type from an interface: `val, ok := i.(T)`

## Q: What is a type switch?
A switch for interface types: `switch v := i.(type) { ... }`

## Q: What is the zero value in Go?
Default value for a type: `0` for ints, `""` for strings, `nil` for pointers/slices/maps.

## Q: Can Go support generics?
Yes, generics were introduced in Go 1.18 using type parameters.

## Q: How do you handle JSON in Go?
Use the `encoding/json` package to marshal and unmarshal.

## Q: What is the difference between `new` and `make`?
`new` allocates memory; `make` initializes slices, maps, or channels.

## Q: How do you do dependency injection in Go?
Manually pass dependencies via constructors. No built-in DI framework.

## Q: What is race condition and how to detect it?
A race condition happens with concurrent writes. Use `go run -race` to detect.

## Q: How do you work with files in Go?
Use `os` and `io/ioutil` or `bufio` packages.

## Q: How do you create custom packages?
Create a directory with a `.go` file and use `package packagename`.

## Q: How do you document code in Go?
Use comments above declarations. `godoc` extracts documentation.

## Q: What are common Go tools?
`go run`, `go build`, `go fmt`, `go test`, `go mod`, `golint`, `go vet`.

## Q: How do you handle concurrency safely?
Use channels and synchronization primitives like mutexes and atomic operations.

## Q: How do you define a method on a type?
Use `func (r ReceiverType) MethodName(args) ReturnType { }`

## Q: What is the difference between a method and a function in Go?
A method is a function with a receiver type.

## Q: How do you implement a stack or queue in Go?
Use slices and custom methods to push/pop or enqueue/dequeue.

## Q: How do you prevent deadlocks in Go?
Avoid blocking calls and use proper synchronization like `select` and `WaitGroup`.

## Q: What are the different ways to loop in Go?
Use `for`, `for range`, and `for` with conditions.

## Q: What is a rune in Go?
A rune is an alias for `int32` and represents a Unicode code point.

## Q: How does Go handle time and date?
Use the `time` package.

## Q: How do you sort a slice in Go?
Use `sort` package: `sort.Ints`, `sort.Strings`, or `sort.Slice` for custom sorting.

## Q: How do you check if a map key exists?
Use value, ok pattern: `val, ok := m[key]`

## Q: How do you convert between types?
Use explicit conversion: `float64(i)`, `string(i)`

## Q: What are tags in struct fields?
Tags are metadata for fields, often used by encoding packages: `json:"fieldName"`

## Q: What is the `iota` identifier in Go?
`iota` is used to simplify constant definitions in blocks.

## Q: What are build tags?
Build tags are special comments that conditionally compile files.

## Q: How do you ignore a test during execution?
Use `t.Skip()` or `// +build !test` directive.

## Q: How do you profile a Go program?
Use `pprof` and `go tool pprof`.

## Q: What is the purpose of `init()` function?
It allows package-level setup before execution.

## Q: How do you embed files in Go?
Use the `embed` package introduced in Go 1.16.

## Q: How do you limit the number of goroutines?
Use buffered channels or worker pools.

## Q: How do you use reflection in Go?
Use the `reflect` package.

## Q: How do you parse command-line arguments?
Use `os.Args` or `flag` package.

## Q: How do you create a web server in Go?
Use `net/http` package with `http.HandleFunc` and `http.ListenAndServe`.

## Q: What is the difference between `panic` and `log.Fatal`?
`panic` triggers a stack trace; `log.Fatal` logs the message and exits.

## Q: How do you chain middleware in Go?
Wrap http.Handler functions.

## Q: How do you mock interfaces in Go?
Use hand-written mocks or libraries like `gomock`, `testify/mock`.

## Q: What is a ticker in Go?
A ticker sends events at intervals. Use `time.NewTicker`.

## Q: What is a timer in Go?
A timer sends a signal after a delay. Use `time.NewTimer`.

## Q: How do you cancel a context?
Use `context.WithCancel()`.

## Q: How do you set timeouts with context?
Use `context.WithTimeout()`.

## Q: How do you retry operations in Go?
Use loops with backoff strategies.

## Q: How do you capture system signals in Go?
Use `os/signal` package.

## Q: How do you use interfaces for testing?
Define behaviors via interfaces and use mocks/stubs.

## Q: How do you avoid circular dependencies?
Refactor code and split packages appropriately.

## Q: How do you use build constraints?
Use `// +build` tags.

## Q: How do you use go:generate?
Add `//go:generate` directives for code generation.

## Q: How do you write concurrent-safe data structures?
Use channels or mutexes properly.

## Q: What is the difference between buffered and unbuffered channels?
Buffered channels don't block until full; unbuffered block until read.

## Q: How do you enforce interface implementation at compile time?
Assign a value to a variable of the interface type.

## Q: How do you implement a plugin system in Go?
Use interfaces and optionally `plugin` package for dynamic loading.

## Q: What is the purpose of the `go.sum` file?
It records cryptographic checksums for modules.

## Q: How do you manage multiple Go versions?
Use `gvm`, `asdf`, or `go install golang.org/dl/go1.X`.

## Q: What is the purpose of `vendor` directory?
It contains local copies of dependencies.

## Q: How do you measure code coverage?
Use `go test -cover` and related flags.

## Q: How do you write custom linters for Go?
Use the `go/ast`, `go/token`, and `go/parser` packages.

## Q: How do you inspect or traverse an AST in Go?
Use `go/parser` and `go/ast`.

## Q: How do you create your own build tools?
Use Go's standard library to script and compile tasks.

## Q: What are common Go community resources?
Go blog, Go Forum, Reddit, StackOverflow, Golang Slack, Awesome Go list.

## Q: What is the role of the Go compiler and linker?
Compiler converts source to binary; linker resolves references and outputs the final binary.

## Q: What is Go (Golang)?
A: Go is a modern programming language created by Google. It's open-source, compiled, and known for being fast, easy to learn, and excellent at handling multiple tasks at once (concurrency). It’s great for building efficient and scalable software.

## Q: What are the key features of Go?
A: Go stands out for its simplicity, speed, and built-in support for concurrency. Key features include:

- Lightweight goroutines for multitasking

- Fast compilation

- Rich standard library

- Garbage collection

- Simple syntax

- Strong typing

- Easy cross-compilation

## Q: How do you declare a variable in Go?
A: You can declare variables in two ways:

With var: var age int = 30

With shorthand (inside functions): age := 30

## Q: What is a goroutine?
A: A goroutine is a lightweight thread managed by Go. It allows you to run functions concurrently using the go keyword:
go fetchData()

## Q: How do channels work in Go?
A: Channels let goroutines communicate with each other safely.
Example:

ch := make(chan int)
ch <- 5           // Send
x := <-ch         // Receive

## Q: What is a slice in Go?
A: A slice is a dynamic, flexible view into an array. You can add, remove, and modify elements easily.
Example:
numbers := []int{1, 2, 3}

## Q: How does Go handle memory management?
A: Go uses automatic garbage collection, so you don’t need to manually free memory. It cleans up unused variables behind the scenes.

## Q: What is the difference between an array and a slice?
A: Arrays have a fixed size, like [5]int. Slices are more flexible and grow as needed, making them the preferred choice in most cases.

## Q: How do you handle errors in Go?
A: Go encourages explicit error handling using return values:
result, err := doSomething()
if err != nil {
    fmt.Println("Error:", err)
}

## Q: What is a struct in Go?
A: A struct is a custom data type that groups fields together:
type Person struct {
    Name string
    Age  int
}
It's like a lightweight class.