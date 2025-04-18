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
