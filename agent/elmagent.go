package agent

import (
	"fmt"
	"math/rand"

	"github.com/infiniteCrank/mathbot/elm"
)

// Environment interface remains the same.
type Environment interface {
	Reset() []float64 // returns initial state
	Step(action int) (nextState []float64, reward float64, done bool)
	ActionSpace() int     // returns number of possible actions
	StateDimensions() int // returns dimensionality of the state vector
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
		// Remove the oldest transition.
		rb.buffer = rb.buffer[1:]
	}
	rb.buffer = append(rb.buffer, t)
}

func (rb *ReplayBuffer) Sample(batchSize int) []Transition {
	// For re-training, we may use all stored transitions.
	if batchSize > len(rb.buffer) {
		batchSize = len(rb.buffer)
	}
	// Randomly sample batchSize elements.
	batch := make([]Transition, batchSize)
	for i := 0; i < batchSize; i++ {
		index := rand.Intn(len(rb.buffer))
		batch[i] = rb.buffer[index]
	}
	return batch
}

// ELMAgent uses an ELM model for decision making.
type ELMAgent struct {
	env          Environment
	model        *elm.ELM
	replayBuffer *ReplayBuffer
	batchSize    int

	gamma         float64 // discount factor
	epsilon       float64 // exploration rate
	minEpsilon    float64
	epsilonDecay  float64
	learnInterval int // number of episodes between re-training updates
}

// NewELMAgent creates a new agent with the given parameters.
func NewELMAgent(env Environment, model *elm.ELM, replayBuffer *ReplayBuffer, batchSize int,
	gamma, epsilon, minEpsilon, epsilonDecay float64, learnInterval int) *ELMAgent {
	return &ELMAgent{
		env:           env,
		model:         model,
		replayBuffer:  replayBuffer,
		batchSize:     batchSize,
		gamma:         gamma,
		epsilon:       epsilon,
		minEpsilon:    minEpsilon,
		epsilonDecay:  epsilonDecay,
		learnInterval: learnInterval,
	}
}

// PredictAction returns an action based on the ELM prediction.
func (a *ELMAgent) PredictAction(state []float64) int {
	qValues := a.model.Predict(state) // ELM.Predict returns a []float64
	return argmax(qValues)
}

// ReTrainModel aggregates training data from the replay buffer and re-trains the ELM.
// Because ELM training is one-shot (closed-form), we re-train over the entire batch.
func (a *ELMAgent) ReTrainModel() {
	if len(a.replayBuffer.buffer) < a.batchSize {
		return // Not enough samples yet.
	}
	// For simplicity, we sample a batch from the buffer.
	minibatch := a.replayBuffer.Sample(a.batchSize)
	var trainStates [][]float64
	var trainTargets [][]float64

	for _, t := range minibatch {
		currentQ := a.model.Predict(t.State)
		target := make([]float64, len(currentQ))
		// Copy current predictions
		copy(target, currentQ)
		// For a one-step episode, if terminal use reward; otherwise, use discounted future value.
		if t.Done {
			target[t.Action] = t.Reward
		} else {
			nextQ := a.model.Predict(t.NextState)
			target[t.Action] = t.Reward + a.gamma*max(nextQ)
		}
		trainStates = append(trainStates, t.State)
		trainTargets = append(trainTargets, target)
	}
	// Re-train the ELM model (closed-form solution using ridge regression)
	a.model.Train(trainStates, trainTargets)
	fmt.Println("Re-trained ELM on", len(trainStates), "samples")
}

// TrainAgent runs episodes, collects experiences, and periodically re-trains the ELM model.
func (a *ELMAgent) TrainAgent(episodes int) {
	for ep := 0; ep < episodes; ep++ {
		state := a.env.Reset()
		done := false
		totalReward := 0.0

		// In our simple environment, episodes are one-step.
		action := 0
		// ε-greedy policy.
		if rand.Float64() < a.epsilon {
			action = rand.Intn(a.env.ActionSpace())
		} else {
			action = a.PredictAction(state)
		}

		nextState, reward, done := a.env.Step(action)
		totalReward += reward

		// Save experience.
		transition := Transition{
			State:     state,
			Action:    action,
			Reward:    reward,
			NextState: nextState,
			Done:      done,
		}
		a.replayBuffer.Add(transition)

		// At the end of each episode, optionally re-train the model.
		if (ep+1)%a.learnInterval == 0 {
			a.ReTrainModel()
		}

		// Decay epsilon.
		if a.epsilon > a.minEpsilon {
			a.epsilon *= a.epsilonDecay
		}

		fmt.Printf("Episode %d, Total Reward: %f, Epsilon: %.4f\n", ep, totalReward, a.epsilon)
	}
}
