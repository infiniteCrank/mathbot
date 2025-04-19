# Error Handling & Best Practices

## Q: How do you define custom error types in Go?
To define custom error types, create a struct and implement the Error method:
```go
type MyError struct {
    Msg string
}

func (e MyError) Error() string {
    return e.Msg
}

func example() error {
    return MyError{Msg: "An error occurred"}
}

```

## Q: How do you wrap errors in Go?
Use the %w verb with fmt.Errorf to wrap an error with additional context:
```go
if err := doSomething(); err != nil {
    return fmt.Errorf("failed to do something: %w", err)
}
```

## Q: How do you handle errors in Go?
In Go, errors are typically handled by checking the returned error value:
```go
if err := someFunction(); err != nil {
    fmt.Println("Error:", err)
}
```

# Go Memory Management

## Q: How does garbage collection work in Go?
Go uses garbage collection (GC) to automatically manage memory by reclaiming unused memory. The GC runs in the background and is concurrent with program execution.

## Q: What is the difference between a pointer and a value in Go?
A value is the actual data, whereas a pointer holds the memory address of a value:
```go
x := 10
p := &x  // p is a pointer to x
fmt.Println(*p)  // Dereferencing the pointer to get the value
```

## Q: How do you allocate memory for slices and maps in Go?
You use make to allocate memory for slices and maps:
```go
slice := make([]int, 0, 10)  // slice with length 0 and capacity 10
m := make(map[string]int)    // map with string keys and int values
```

# Go Performance and Optimization

## Q: How do you measure execution time in Go?
```go
start := time.Now()
doWork()
fmt.Println("Execution time:", time.Since(start))
```

## Q: How do you handle concurrency issues in Go?
Concurrency issues, such as race conditions, are often handled by:
1. Using sync.Mutex for locking shared resources.
2. Using sync.WaitGroup to wait for goroutines to finish.
3. Avoiding shared state by using channels to communicate between goroutines.

## Q: How do you optimize performance in Go?
You can optimize performance in Go by:
1. Using efficient data structures like map and slice.
2. Avoiding unnecessary allocations.
3. Using sync.Pool for reusing objects.
4. Minimizing locks and using fine-grained concurrency.

# Miscellaneous

## Q: How do you format code in Go?
Use the built-in gofmt tool to automatically format your Go code:
```bash
gofmt -w yourfile.go
```

## Q: How do you run tests in Go?
Use the go test command to run tests in the current directory:
```bash
go test
```

## Q: How do you implement a queue in Go?
A simple queue can be implemented using a slice:
```go
type Queue []int

func (q *Queue) Enqueue(val int) {
    *q = append(*q, val)
}

func (q *Queue) Dequeue() int {
    val := (*q)[0]
    *q = (*q)[1:]
    return val
}
```

# Go Language Features

## Q: How do you define and use constants in Go?
You define constants using the const keyword:
```go
const Pi = 3.14159
const Hello = "Hello, world!"
```

## Q: How do you create and use a struct in Go?
A struct is a composite data type that groups together variables of different types:
```go
type Person struct {
    Name  string
    Age   int
}

p := Person{Name: "John", Age: 30}
fmt.Println(p.Name) // Output: John
```

## Q: What is the init() function in Go?
The init() function is executed automatically before main() and is typically used for initialization tasks, like setting up variables or establishing connections:
```go
func init() {
    fmt.Println("Initialization tasks")
}
```

## Q: What are Go's built-in data types?
Go has a variety of built-in data types, including:
- Numeric types: int, float32, float64, etc.
- Boolean: bool
- String: string
- Composite types: array, slice, map, struct, channel
- Pointer: *T (a pointer to type T)

## Q: How do you use the defer statement in Go?
The defer statement schedules a function to be executed once the surrounding function returns. It's commonly used for cleanup tasks:
```go
defer fmt.Println("This will run last")
fmt.Println("This will run first")
```

# Go Concurrency

## Q: What is a goroutine scheduler?
The Go runtime scheduler manages the execution of goroutines, multiplexing them onto available threads. It schedules goroutines based on availability and workload.

## Q: How do you handle panics in goroutines?
You can recover from panics in a goroutine by using defer and recover:
```go
go func() {
    defer func() {
        if r := recover(); r != nil {
            fmt.Println("Recovered from panic:", r)
        }
    }()
    // code that might panic
}()
```

## Q: What is the difference between buffered and unbuffered channels?
- Buffered channels have a capacity and allow sending/receiving values without blocking until the buffer is full/empty.
- Unbuffered channels block the sender and receiver until the other side is ready.

## Q: How do you synchronize goroutines?
You can synchronize goroutines using:
1. sync.WaitGroup for waiting until all goroutines complete.
2. sync.Mutex for locking shared resources.
3. sync.Cond for signaling between goroutines.

## Q: How do you manage goroutine lifecycle?
You can manage the lifecycle by controlling when they start, wait for completion, and signal termination using channels or a sync.WaitGroup.

## Q: What is a goroutine?
A goroutine is a lightweight thread managed by the Go runtime. You start one by prefixing a function call with `go`:
```go
go doWork()
```

## Q: What is a channel?
A channel is a typed conduit through which you can send and receive values with the channel operator `<-`:
```go
ch := make(chan int)
go func() {
    ch <- 5  // send 5 to the channel
}()
value := <-ch  // receive from channel
```

## Q: How do you use `select` with channels?
`select` lets you wait on multiple channel operations:
```go
select {
case v := <-ch1:
    fmt.Println("received", v, "from ch1")
case ch2 <- 3:
    fmt.Println("sent 3 to ch2")
default:
    fmt.Println("no communication")
}
```

## Q: What are buffered channels?
Buffered channels have a capacity and don’t block until full:
```go
ch := make(chan int, 2)
ch <- 1
ch <- 2
```

## Q: How do you aggregate errors from concurrent tasks?
Use a mutex and slice to collect errors safely across goroutines:
```go
var wg sync.WaitGroup
var mu sync.Mutex
var errors []error

for _, task := range tasks {
    wg.Add(1)
    go func(t Task) {
        defer wg.Done()
        if err := t.Execute(); err != nil {
            mu.Lock()
            errors = append(errors, err)
            mu.Unlock()
        }
    }(task)
}
wg.Wait()

if len(errors) > 0 {
    // Handle aggregate errors
}
```

## Q: How do you handle cancellation and timeouts in goroutines?
Use the `context.Context` package for propagating cancelation signals:
```go
ctx, cancel := context.WithTimeout(context.Background(), time.Second)
defer cancel()

// Use ctx in a goroutine
go func(ctx context.Context) {
    select {
    case <-ctx.Done():
        fmt.Println("Operation timed out")
    case result := <-performLongOperation(ctx):
        fmt.Println("Result:", result)
    }
}(ctx)
```

## Q: How do you implement graceful shutdown?
Use a signal handler to catch termination signals and shut down your server cleanly:
```go
func shutdownServer(s *http.Server) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    if err := s.Shutdown(ctx); err != nil {
        log.Fatal("Server forced to shutdown:", err)
    }
}

// Signal handling for graceful shutdown
c := make(chan os.Signal, 1)
signal.Notify(c, os.Interrupt)

go func() {
    <-c
    shutdownServer(s)
}()
```

# Go Language Features

## Q: What are variadic functions in Go?
Functions that accept a variable number of arguments:
```go
func Sum(numbers ...int) int {
    total := 0
    for _, n := range numbers {
        total += n
    }
    return total
}
```

## Q: What is a blank identifier?
Used to ignore a value returned from a function:
```go
_, err := someFunc()
```

## Q: What are zero values in Go?
Default values for variables when they are declared without initialization:
- Numeric types: 0
- Strings: ""
- Booleans: false
- Pointers/interfaces: nil

## Q: How does defer work?
Schedules a function call to run after the function completes:
```go
file, _ := os.Open("file.txt")
defer file.Close()
```

## Q: How do you write table-driven tests in Go?
```go
tests := []struct{
    input int
    want int
}{
    {1, 2},
    {2, 4},
}

for _, tc := range tests {
    t.Run(fmt.Sprintf("input: %d", tc.input), func(t *testing.T) {
        got := double(tc.input)
        if got != tc.want {
            t.Errorf("got %d, want %d", got, tc.want)
        }
    })
}
```

## Q: What is the functional options pattern?
A flexible way to configure objects with optional parameters:
```go
type Server struct {
    port int
    timeout time.Duration
}

type Option func(*Server)

func WithPort(port int) Option {
    return func(s *Server) { s.port = port }
}

func NewServer(opts ...Option) *Server {
    s := &Server{port: 8080}
    for _, opt := range opts {
        opt(s)
    }
    return s
}
```

## Q: How do interfaces work in Go?
Interfaces define method sets. Any type that implements those methods satisfies the interface:
```go
type Shape interface {
    Area() float64
}

type Square struct { Side float64 }
func (s Square) Area() float64 { return s.Side * s.Side }
```

## Q: What is struct embedding?
Allows a struct to include another, inheriting its fields/methods:
```go
type Person struct { Name string }
type Employee struct { Person; ID int }
```

## Q: How do you use the `reflect` package?
For inspecting and manipulating types/values at runtime:
```go
x := 42
t := reflect.TypeOf(x)
v := reflect.ValueOf(x)
```

## Q: How do you use WaitGroups?
Wait for a collection of goroutines to finish:
```go
var wg sync.WaitGroup
wg.Add(1)
go func() {
    defer wg.Done()
    // work
}()
wg.Wait()
```

## Q: What is the command pattern?
Encapsulates an action as an object:
```go
type Command interface { Execute() }
type ConcreteCommand struct { }
func (c ConcreteCommand) Execute() { fmt.Println("Executing") }
```

## Q: How do you define and use custom error types?
```go
type MyError struct { Msg string }
func (e MyError) Error() string { return e.Msg }
```

## Q: How do you wrap errors with context?
Use `fmt.Errorf` with `%w`:
```go
if err := doThing(); err != nil {
    return fmt.Errorf("doThing failed: %w", err)
}
```

## Q: How do you close channels properly?
Close them once no more values will be sent:
```go
close(ch)
```

## Q: How do you manage long-running goroutines?
Use a `done` channel to stop them:
```go
done := make(chan struct{})
go func() {
    for {
        select {
        case <-done:
            return
        }
    }
}()
```

## Q: What is the purpose of init() functions?
They run before the `main()` function and are useful for initialization logic.

# More Go Questions and Answers

## Q: How do you declare a constant in Go?
Use the `const` keyword:
```go
const Pi = 3.14
```

## Q: What is a rune in Go?
A rune is an alias for `int32` and is used to represent a Unicode code point:
```go
var r rune = '♥'
```

## Q: How do you convert a string to an int?
Use `strconv.Atoi`:
```go
num, err := strconv.Atoi("42")
```

## Q: How do you handle panics?
Use `recover` in a deferred function:
```go
defer func() {
    if r := recover(); r != nil {
        fmt.Println("Recovered from panic:", r)
    }
}()
```

## Q: How do you iterate over a map?
```go
m := map[string]int{"a": 1, "b": 2}
for k, v := range m {
    fmt.Println(k, v)
}
```

## Q: How do you implement a method for a struct?
```go
type Person struct { Name string }
func (p Person) Greet() string {
    return "Hello, " + p.Name
}
```

## Q: How do you create and use a slice?
```go
s := []int{1, 2, 3}
s = append(s, 4)
```

## Q: What is the difference between make and new?
- `make` is used for slices, maps, and channels.
- `new` allocates memory and returns a pointer.

## Q: How do you check if a map key exists?
```go
val, ok := myMap[key]
```

## Q: What is the syntax for defining a function in Go?
To define a function in Go, the syntax would be:
```go
func Addition() {
  var a = 2
  var b = 2
  c:= a+b
}
```

## Q: How do you define a function in Go with parameters?
To define a function with parameters in Go, you would use the following syntax:
```go
func Addition(a,b int) {
  c:= a+b
}
```

## Q: How do you define a function in Go that returns a value?
To define a function that returns a value in Go, the syntax is:
```go
func Addition(a,b int) int{
  c:= a+b
  return c
}
```

## Q: How do you define a function in Go with multiple return values?
To define a function with multiple return values in Go, use the following syntax:
```go
func Divide(a,b int) (int,error){
  if b < 1 {
    return fmt.Errorf("Divide by zero"), nil
  }
  c:= a/b
  return c, nil
}
```

## Q: What are some key features of the Go programming language?
Go is designed for system programming and emphasizes simplicity and efficiency. Key features include:
- Static Typing: Strong and explicit variable types that improve error detection at compile time.
- Garbage Collection: Automatic memory management that helps prevent memory leaks.
- Concurrency: Built-in support for concurrent programming with go-routines and channels.

## Q: What are the principles of writing effective Go code?
Key principles include:
- Naming Conventions
    - Choose clear and descriptive names for variables, functions, and types.
    - Use mixedCaps or CamelCase for multi-word names.
    - Avoid acronyms unless they are widely known.
    - Start files with the name of the package they belong to.
- Error Handling
    - Go does not have exceptions. Error handling is managed by returning error values.
    ```go 
    result, err := someOperation()
    if err != nil {
        log.Fatal(err)
    }
    ```
    - Consider wrapping errors with additional context using fmt.Errorf.
    ```go
    return fmt.Errorf("failed to process data: %w", err)
    ```
- Code Formatting
    - Consistently format code using gofmt, which is included in every Go installation.
- Control Structures
    - Use if, for, and switch as your primary control flow mechanisms. Go does not have a while statement; use for with conditions instead.
    ```go
    for i := 0; i < 10; i++ {
        fmt.Println(i)
    }
    ```
- Defer, Panic, and Recover
    - defer statements are executed after the function returns:
    ```go 
    defer fmt.Println("World")
    fmt.Println("Hello")
    ```
    - Use panic for unrecoverable errors and recover to regain control in deferred functions.

## Q: How do you create a package in Go?
To create a package, define a package at the top of each file:
```go 
package mypackage
```

## Q: How do you import packages in Go?
Import standard and custom packages using the import statement:
```go
import (
    "fmt"
    "myapp/mypackage"
)
```

## Q: What is concurrency in Go?
Concurrency is one of Go's standout features, allowing multiple tasks to progress at once. Key constructs include go-routines and channels.

## Q: How do you start a new go-routine?
Start a new go-routine with the go keyword:
```go
go myFunction()
```

## Q: How are channels used in Go?
Channels are used to synchronize and communicate between go-routines:
```go 
ch := make(chan int)
go func() {
    ch <- 42 // Sending value
}()
value := <-ch // Receiving value
```

## Q: What are buffered channels in Go?
Buffered channels can store a limited number of messages:
```go
ch := make(chan int, 2) // Buffered channel with capacity 2
ch <- 1
ch <- 2
```

## Q: How do you use the select statement in Go?
Use the select statement to wait on multiple channel operations:
```go
select {
case msg := <-ch1:
    fmt.Println("Received:", msg)
case msg := <-ch2:
    fmt.Println("Received:", msg)
case <-time.After(time.Second):
    fmt.Println("Timeout")
}
```

## Q: What does the Go standard library offer?
The Go standard library offers a wide range of functionality across various domains.

## Q: What is the net/http package used for in Go?
The net/http package is used for establishing HTTP servers and clients:
```go
http.HandleFunc("/", handler)
http.ListenAndServe(":8080", nil)
```

## Q: How can you work with JSON in Go?
Use the encoding/json package for JSON encoding and decoding:
```go
type Person struct {
    Name string `json:"name"`
    Age  int    `json:"age"`
}

person := Person{Name: "Alice", Age: 30}

// Encoding a struct to JSON
jsonData, err := json.Marshal(person)
if err != nil {
    log.Fatal(err)
}
fmt.Println(string(jsonData))  // Output: {"name":"Alice","age":30}

// Decoding JSON back to struct
var decodedPerson Person
jsonString := `{"name":"Bob", "age":25}`
err = json.Unmarshal([]byte(jsonString), &decodedPerson)
if err != nil {
    log.Fatal(err)
}
fmt.Println(decodedPerson)  // Output: {Bob 25}
```

## Q: How do you set up database connections in Go?
Open a connection to a database using sql.Open:
```go
import (
    "database/sql"
    _ "GitHub.com/go-sql-driver/mysql" // MySQL driver
)

db, err := sql.Open("mysql", "user:password@tcp(127.0.0.1:3306)/myDb")
if err != nil {
    log.Fatal(err)
}
defer db.Close()
```

## Q: How do you execute queries in Go?
You can use Exec, Query, and QueryRow for executing SQL commands:
```go
// Insert a row
result, err := db.Exec("INSERT INTO users(name, age) VALUES(?, ?)", "John Doe", 30)

// Querying rows
rows, err := db.Query("SELECT name, age FROM users")
defer rows.Close()
for rows.Next() {
    var name string
    var age int
    if err := rows.Scan(&name, &age); err != nil {
        log.Fatal(err)
    }
    fmt.Println(name, age)
}
```

## Q: How can you handle file I/O in Go?
Read from and write to files using the os and io/ioutil packages:
```go
data, err := ioutil.ReadFile("input.txt")
err = ioutil.WriteFile("output.txt", data, 0644)
```

## Q: What is the Go development and release process like?
Go has a well-structured development process with a regular release schedule that includes improvements, features, and bug fixes:
- Major versions are released approximately every six months, with minor updates in between.
- Each release is accompanied by comprehensive release notes detailing new features, deprecated features, and bug fixes.
- Developers are encouraged to review release notes to stay updated on new functionalities and changes.

## Q: How do you update Go and its packages?
To ensure the best performance, feature set, and security:
1. Use go get to update: Keep your Go installation and packages updated with the latest versions.
```go 
go get -u all
```
2. Review Release Notes: Check the Release Notes before upgrading.

## Q: How do you perform testing in Go?
Testing is essential for ensuring code correctness. Go offers a built-in testing framework.

## Q: How do you write tests in Go?
Place your tests in files named *_test.go. A simple test function looks like this:
```go 
package mypackage

import "testing"

func TestAddition(t *testing.T) {
    result := add(1, 2)
    expected := 3
    if result != expected {
        t.Errorf("Expected %d, but got %d", expected, result)
    }
}
```

## Q: How do you run tests in Go?
Use the go test command to run tests:
```go 
go test
```

## Q: How do you create benchmark tests in Go?
Benchmark functions can be created by starting the function name with Benchmark:
```go 
func BenchmarkAddition(b *testing.B) {
    for i := 0; i < b.N; i++ {
        add(1, 2)
    }
}
```

## How do you measure test coverage in Go?
You can measure test coverage with:
```go
go test -cover
```

## Q: What are the documentation standards in Go?
Proper documentation is crucial for maintaining and understanding code.

## Q: How do you write documentation for packages in Go?
Each package should include a doc.go file that describes its purpose and usage. Functions and types should have comments that explain their behavior and parameters.

## Q: How does GoDoc work?
Use // comments directly before a function or type definition for GoDoc to parse and generate documentation

## Q: How do you generate documentation for your Go package?
You can generate documentation for your package using:
```go
go doc
go doc mypackage.FunctionName
```

## Q: What does the sync package provide in Go?
The sync package provides synchronization primitives such as mutexes and wait groups:
```go
import "sync"

var mu sync.Mutex

// Protect shared data with a mutex.
mu.Lock()
// Perform operations on shared data.
mu.Unlock()
```

## Q: What capabilities does the time package offer?
The time package encompasses time manipulation and measurement:
```go
import "time"

// Get the current time
now := time.Now()

// Sleep for 2 seconds
time.Sleep(2 * time.Second)
```

## Q: How does the http package function in Go?
The http package has built-in support for creating HTTP servers and clients:
```go 
import (
    "net/http"
)

http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
    w.Write([]byte("Hello, Go!"))
})

http.ListenAndServe(":8080", nil)
```

## Q: What is the context package used for in Go?
The context package provides a way to pass deadlines, cancellation signals, and request-scoped values across API boundaries and go-routines:
```go 
import "context"

// Create a context with a timeout
ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
defer cancel()

// Use ctx in a function
func(ctx context.Context) {
}

// Use ctx with a goroutine
go func(ctx context.Context) {
    select {
    case <-ctx.Done():
        fmt.Println("Operation cancelled:", ctx.Err())
        return
    case result := <-performOperation(ctx):
        fmt.Println("Received result:", result)
    }
}(ctx)

// Perform an operation that honors the passed context
func performOperation(ctx context.Context) <-chan int {
    ch := make(chan int)
    go func() {
        // Simulate work
        time.Sleep(1 * time.Second) // Could be interrupted
        ch <- 42                     // Send result on channel
    }()
    return ch
}
```

## Q: How does Go handle networking with the net package?
Go makes it easy to perform networking tasks with the net package.

## How can you create a simple TCP client in Go?
Here is a simple TCP client that connects to a server and sends a message:
```go
import (
    "net"
    "fmt"
    "os"
)

func main() {
    conn, err := net.Dial("tcp", "localhost:8080")
    if err != nil {
        fmt.Println("Error connecting:", err)
        os.Exit(1)
    }
    defer conn.Close()

    _, err = conn.Write([]byte("Hello Server!"))
    if err != nil {
        fmt.Println("Error writing to server:", err)
    }
}
```

## Q; How to set up a simple TCP server in Go?
To set up a simple TCP server that listens for connections:

```go
import (
    "net"
    "fmt"
)

func main() {
    listener, err := net.Listen("tcp", ":8080")
    if err != nil {
        fmt.Println("Error starting TCP server:", err)
        return
    }
    defer listener.Close()

    for {
        conn, err := listener.Accept()
        if err != nil {
            fmt.Println("Error accepting connection:", err)
            continue
        }
        go handleConnection(conn) // Handle each connection in a new goroutine
    }
}

func handleConnection(conn net.Conn) {
    defer conn.Close()
    buffer := make([]byte, 1024)
    _, err := conn.Read(buffer)
    if err != nil {
        fmt.Println("Error reading from connection:", err)
        return
    }
    fmt.Println("Received:", string(buffer))
}
```

## Q: What are some common patterns and best practices in Go?
Efficient Go programming is often about following established patterns and practices.

## Q: How can you handle configuration effectively?
Use a configuration struct to manage application settings in one place:
```go
type Config struct {
    Port        int
    DatabaseURL string
}

func LoadConfig() Config {
    return Config{Port: 8080, DatabaseURL: "user:password@tcp(127.0.0.1:3306)/mydb"}
}
```

## Q: What is graceful shutdown in Go?
Implement graceful shutdown for applications to allow in-progress requests to complete:
```go
func shutdownServer(s *http.Server) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    if err := s.Shutdown(ctx); err != nil {
        log.Fatal("Server forced to shutdown:", err)
    }
}

// Signal handling for graceful shutdown
c
```

## Q: How can you implement signal handling for graceful shutdown?
You can implement signal handling for graceful shutdown like this:
```go
c := make(chan os.Signal, 1)
signal.Notify(c, os.Interrupt)

go func() {
    <-c
    shutdownServer(s)
}()
```

## Q: What are common idioms in Go programming?
Go programming has several idioms that developers use to improve clarity and maintainability.

## How do you define and use variadic functions?
Variadic functions allow handling a variable number of parameters:
```go
func Sum(numbers ...int) int {
    total := 0
    for _, n := range numbers {
        total += n
    }
    return total
}

// Call with any number of arguments
result := Sum(1, 2, 3, 4, 5)
fmt.Println("Total:", result) // Output: Total: 15
```

## Q: How do you create slices and maps in Go?
- Creating a slice:
```go
numbers := []int{1, 2, 3} // Slice of integers
```
- Creating a map:
```go
ages := map[string]int{
    "Alice": 30,
    "Bob":   25,
}
```

## Q: How is string formatting handled in Go?
Go provides a robust way to format strings:
```go 
msg := fmt.Sprintf("User %s has %d points.", username, points)
```

## Q: What is the blank identifier used for in Go?
You can use the blank identifier when you want to ignore one of the returned values:
```go
_, err := someFunction() // Ignore the first return value
```

## Q: How do you handle errors early in Go?
A common pattern in Go is to return quickly on error checks:
```go
if err != nil {
    return err // Early return on error
}
```

## Q: How do you check for errors idiomatically in Go?
Check for errors immediately after a function call in this way:
```go
if err != nil {
    return err // Handle the error immediately
}
```

## Q: How do you synchronize goroutines with WaitGroups?
The sync.WaitGroup allows you to wait for a group of goroutines to finish executing:
```go
var wg sync.WaitGroup
wg.Add(1)
go func() {
    defer wg.Done()
    // Perform work
}()
wg.Wait() // Wait until all goroutines have completed
```

## Q: What is the purpose of init functions in Go?
init functions are executed before the main function, allowing for package-level initialization.

## Q: What are zero values in Go?
All Go types have a default zero value:
- Numeric types: 0
- Booleans: false
- Strings: ""
- Pointers: nil

## Q: How can you defer cleanup actions in Go?
Use defer for clean-up actions to ensure resources are freed properly:
```go 
file, err := os.Open("example.txt")
if err != nil {
    log.Fatal(err)
}
defer file.Close() // Ensures that the file is closed when the function exits
```

## Q: How is resource management handled in Go?
Proper resource management in Go is vital for writing efficient and bug-free applications.

## Q: How do you manage goroutines effectively?
Ensure goroutines terminate when they are no longer needed. Use channels to signal when to stop:
```go
done := make(chan struct{})
go func() {
    for {
        select {
        case <-done:
            return // Exit goroutine when done is closed
        default:
            // Perform work
        }
    }
}()
```

## Q: Why should you close channels in Go?
Always close channels when they are no longer needed to prevent resource leaks and ensure proper synchronization:
```go
close(ch) // Close the channel when done sending data
```

## Q: What are testing patterns in Go?
Adopting correct patterns in testing can enhance the quality and reliability of your code.

## Q: What is the table-driven test pattern in Go?
This pattern is a common approach in Go to run multiple tests using the same function:
```go
type test struct {
    input    int
    expected int
}

tests := []test{
    {1, 2},
    {2, 4},
}

for _, tc := range tests {
    t.Run(fmt.Sprintf("Input: %d", tc.input), func(t *testing.T) {
        got := double(tc.input)
        if got != tc.expected {
            t.Errorf("Expected %d, got %d", tc.expected, got)
        }
    })
}
```

## Q: How do you use dependency injection in Go?
You can inject dependencies through function arguments to enhance testability:
```go
type Repository interface {
    Save(data string)
}

func SaveData(repo Repository, data string) {
    repo.Save(data)
}
```

## Q: What error handling approaches are effective in Go?
Error handling is a fundamental aspect of Go programming.

## Q: How can you create custom error types in Go?
Implement custom error types to provide more context:
```go
type MyError struct {
    Msg string
}

func (e MyError) Error() string {
    return e.Msg
}

func doSomething() error {
    return MyError{"Something went wrong!"}
}
```

## Q: How can errors be wrapped in Go?
Use the fmt.Errorf function to wrap errors with additional context:
```go
func doTask() error {
    if err := taskThatMightFail(); err != nil {
        return fmt.Errorf("doTask failed: %w", err)
    }
    return nil
}
```

## Q: How do you handle multiple errors concurrently?
Aggregate errors when running multiple concurrent tasks:
```go
var wg sync.WaitGroup
var mu sync.Mutex
var errors []error

for _, task := range tasks {
    wg.Add(1)
    go func(t Task) {
        defer wg.Done()
        if err := t.Execute(); err != nil {
            mu.Lock()
            errors = append(errors, err)
            mu.Unlock()
        }
    }(task)
}
wg.Wait()

if len(errors) > 0 {
    // Handle aggregate errors
}
```

## Q: What are common patterns in Go programming?
Understanding common patterns can help in writing clean, maintainable Go code.

## Q: How does the functional options pattern work?
This pattern allows configurations of struct initialization by passing options:
```go
type Server struct {
    port int
    timeout time.Duration
}

type Option func(*Server)

func WithPort(port int) Option {
    return func(s *Server) {
        s.port = port
    }
}

func WithTimeout(t time.Duration) Option {
    return func(s *Server) {
        s.timeout = t
    }
}

func NewServer(opts ...Option) *Server {
    s := &Server{
        port: 8080, // default port
        timeout: time.Second * 5, // default timeout
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}
```

## Q: What is the command pattern in Go?
The command pattern encapsulates requests as objects to allow parameterization of clients with queues, requests, and operations:
```go
type Command interface {
    Execute()
}

type ConcreteCommand struct {
    receiver *Receiver
}

func (c *ConcreteCommand) Execute() {
    c.receiver.Action()
}
```

## Q: How is context and cancellation handled in Go?
Use the context.Context for managing deadlines and cancellation:
```go
ctx, cancel := context.WithTimeout(context.Background(), time.Second)
defer cancel()

// Use ctx in a goroutine
go func(ctx context.Context) {
    select {
    case <-ctx.Done():
        fmt.Println("Operation timed out")
    case result := <-performLongOperation(ctx):
        fmt.Println("Result:", result)
    }
}(ctx)
```
