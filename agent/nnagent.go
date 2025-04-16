package agent

import (
	"fmt"
	"math/rand"

	"github.com/infiniteCrank/mathbot/NeuralNetwork"
	_ "github.com/lib/pq" // PostgreSQL driver if you use PostgreSQL.
)

// ---------- New Self-Learning Components ----------

// Environment interface defines the methods an environment must implement.
type Environment interface {
	Reset() []float64 // returns initial state
	Step(action int) (nextState []float64, reward float64, done bool)
	ActionSpace() int     // returns number of possible actions
	StateDimensions() int // returns dimensionality of the state vector
}

// A simple sample environment that generates random states and rewards.
// Replace this with your own environment logic (e.g., a game or control simulation).
type RandomEnv struct {
	stateDim    int
	actionSpace int
	stepCount   int
	maxSteps    int
}

func NewRandomEnv(stateDim, actionSpace, maxSteps int) *RandomEnv {
	return &RandomEnv{
		stateDim:    stateDim,
		actionSpace: actionSpace,
		maxSteps:    maxSteps,
	}
}

func (env *RandomEnv) Reset() []float64 {
	env.stepCount = 0
	state := make([]float64, env.stateDim)
	for i := 0; i < env.stateDim; i++ {
		state[i] = rand.Float64() // random initial state
	}
	return state
}

func (env *RandomEnv) Step(action int) (nextState []float64, reward float64, done bool) {
	env.stepCount++
	nextState = make([]float64, env.stateDim)
	for i := 0; i < env.stateDim; i++ {
		nextState[i] = rand.Float64()
	}
	// For demonstration, reward is a random value (or could depend on action)
	reward = rand.Float64() * 10.0
	if env.stepCount >= env.maxSteps {
		done = true
	}
	return
}

func (env *RandomEnv) ActionSpace() int {
	return env.actionSpace
}

func (env *RandomEnv) StateDimensions() int {
	return env.stateDim
}

// Transition holds one experience tuple.
type Transition struct {
	State     []float64
	Action    int
	Reward    float64
	NextState []float64
	Done      bool
}

// ReplayBuffer stores a fixed number of transitions.
type ReplayBuffer struct {
	buffer  []Transition
	maxSize int
}

func NewReplayBuffer(size int) *ReplayBuffer {
	return &ReplayBuffer{
		maxSize: size,
		buffer:  make([]Transition, 0, size),
	}
}

func (rb *ReplayBuffer) Add(t Transition) {
	if len(rb.buffer) >= rb.maxSize {
		// remove the oldest transition
		rb.buffer = rb.buffer[1:]
	}
	rb.buffer = append(rb.buffer, t)
}

func (rb *ReplayBuffer) Sample(batchSize int) []Transition {
	batch := make([]Transition, batchSize)
	for i := 0; i < batchSize; i++ {
		index := rand.Intn(len(rb.buffer))
		batch[i] = rb.buffer[index]
	}
	return batch
}

// ---------- Agent that uses the network to learn from interactions ----------

type Agent struct {
	env           Environment
	network       *NeuralNetwork.NeuralNetwork
	replayBuffer  *ReplayBuffer
	batchSize     int
	gamma         float64 // discount factor
	epsilon       float64 // exploration rate
	minEpsilon    float64
	epsilonDecay  float64
	learnInterval int // number of steps between learning updates
}

func NewAgent(env Environment, network *NeuralNetwork.NeuralNetwork, replayBuffer *ReplayBuffer, batchSize int,
	gamma float64, epsilon float64, minEpsilon float64, epsilonDecay float64, learnInterval int) *Agent {
	return &Agent{
		env:           env,
		network:       network,
		replayBuffer:  replayBuffer,
		batchSize:     batchSize,
		gamma:         gamma,
		epsilon:       epsilon,
		minEpsilon:    minEpsilon,
		epsilonDecay:  epsilonDecay,
		learnInterval: learnInterval,
	}
}

// argmax returns the index of the maximum value in a slice.
func argmax(values []float64) int {
	maxIndex := 0
	maxVal := values[0]
	for i, v := range values {
		if v > maxVal {
			maxVal = v
			maxIndex = i
		}
	}
	return maxIndex
}

// max returns the maximum value in a slice.
func max(values []float64) float64 {
	maxVal := values[0]
	for _, v := range values {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

// TrainAgent runs episodes, collects experiences, and updates the network.
func (a *Agent) TrainAgent(episodes int) {
	stepCount := 0
	for ep := 0; ep < episodes; ep++ {
		state := a.env.Reset()
		done := false
		totalReward := 0.0

		for !done {
			var action int
			// ε-greedy policy: explore or exploit.
			if rand.Float64() < a.epsilon {
				action = rand.Intn(a.env.ActionSpace())
			} else {
				qValues := a.network.PredictRegression(state)
				action = argmax(qValues)
			}

			nextState, reward, done := a.env.Step(action)
			totalReward += reward

			// Save transition.
			transition := Transition{
				State:     state,
				Action:    action,
				Reward:    reward,
				NextState: nextState,
				Done:      done,
			}
			a.replayBuffer.Add(transition)
			state = nextState
			stepCount++

			// Update network after fixed interval and if enough samples exist.
			if len(a.replayBuffer.buffer) >= a.batchSize && stepCount%a.learnInterval == 0 {
				minibatch := a.replayBuffer.Sample(a.batchSize)
				trainStates := [][]float64{}
				targets := [][]float64{}

				for _, t := range minibatch {
					currentQ := a.network.PredictRegression(t.State)
					target := make([]float64, len(currentQ))
					copy(target, currentQ)
					// If terminal state, target is just the reward.
					if t.Done {
						target[t.Action] = t.Reward
					} else {
						nextQ := a.network.PredictRegression(t.NextState)
						target[t.Action] = t.Reward + a.gamma*max(nextQ)
					}
					trainStates = append(trainStates, t.State)
					targets = append(targets, target)
				}
				// A single update iteration with no decay for demonstration.
				a.network.Train(trainStates, targets, 1, 1.0, 1, a.batchSize)
			}
		} // end episode

		// Decay epsilon.
		if a.epsilon > a.minEpsilon {
			a.epsilon *= a.epsilonDecay
		}
		fmt.Printf("Episode %d, Total Reward: %f, Epsilon: %.4f\n", ep, totalReward, a.epsilon)
	}
}
