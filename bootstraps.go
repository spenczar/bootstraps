package bootstraps

import (
	"encoding/binary"
	"math/rand"
)

// A Bootstrapper is capable of performing bootstrap calculations.
type Bootstrapper struct {
	rand *rand.Rand
}

// New creates a new Bootstrapper using the provided random number source.
func New(src rand.Source64) Bootstrapper {
	return Bootstrapper{
		rand: rand.New(src),
	}
}

func (b Bootstrapper) randIndexes(maxVal uint64, n uint64) []int {
	out := make([]int, n)
	randUints := make([]uint64, n)
	binary.Read(b.rand, binary.LittleEndian, randUints)
	for i, v := range randUints {
		out[i] = int(v % n)
	}
	return out
}

// Mean computes the bootstrapped mean distribution of the dataset.
func (b Bootstrapper) Mean(data []float64, samples int) []float64 {
	means := make([]float64, samples)
	n := uint64(len(data))
	for i := 0; i < samples; i++ {
		idxes := b.randIndexes(n, n)
		for j, idx := range idxes {
			means[i] = ((means[i] * float64(j)) + data[idx]) / float64(j+1)
		}
	}
	return means
}

// Bootstrap applies an arbitrary function to resampled copies of the dataset.
// It returns a slice of the output function values.
func (b Bootstrapper) Bootstrap(data []float64, samples int, f func([]float64) float64) []float64 {
	vals := make([]float64, samples)
	n := uint64(len(data))
	for i := 0; i < samples; i++ {
		idxes := b.randIndexes(n, n)
		sample := make([]float64, int(n))
		for j, idx := range idxes {
			sample[j] = data[idx]
		}
		vals[i] = f(sample)
	}
	return vals
}

// StreamingBootstrap applies a streaming (or 'online') function to resampled
// values in the dataset. It returns a slice of the output function values.
// Streaming can be much more memory efficient than Bootstrap, but not all
// functions of a dataset can be expressed in terms of a streaming function.
func (b Bootstrapper) StreamingBootstrap(data []float64, samples int, f StreamingFunc) []float64 {
	vals := make([]float64, samples)
	n := uint64(len(data))
	for i := 0; i < samples; i++ {
		idxes := b.randIndexes(n, n)
		for j, idx := range idxes {
			vals[i] = f(j, vals[i], data[idx])
		}
	}
	return vals
}

// A StreamingFunc computes a streaming value. i represents the number of values
// seen so far in this streaming computation (0-indexed). prev is the last value
// returned from the StreamingFunc, and val is the input value.
//
// See StreamingMean's implementation for a concrete example, as it implements
// the mean through a streaming algorithm.
type StreamingFunc func(i int, prev, val float64) float64

var StreamingMean StreamingFunc = func(i int, prev, val float64) float64 {
	return ((prev * float64(i)) + val) / float64(i+1)
}
