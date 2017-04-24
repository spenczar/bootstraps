package bootstraps

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMean(t *testing.T) {
	s := rand.NewSource(1).(rand.Source64)
	bs := New(s)
	data := make([]float64, 1000)
	for i := range data {
		data[i] = float64(i)
	}

	have := bs.Mean(data, 10000)
	want := trueMean(data)

	haveMean := trueMean(have)
	assert.InEpsilon(t, want, haveMean, 0.001)

	for _, v := range have {
		assert.InEpsilon(t, want, v, 0.1)
	}
}

func TestStreamingBootstrop(t *testing.T) {
	s := rand.NewSource(1).(rand.Source64)
	bs := New(s)
	data := make([]float64, 1000)
	for i := range data {
		data[i] = float64(i)
	}

	have := bs.StreamingBootstrap(data, 10000, StreamingMean)
	want := trueMean(data)

	haveMean := trueMean(have)
	assert.InEpsilon(t, want, haveMean, 0.001)

	for _, v := range have {
		assert.InEpsilon(t, want, v, 0.1)
	}
}

func TestBootstrap(t *testing.T) {
	s := rand.NewSource(1).(rand.Source64)
	bs := New(s)
	data := make([]float64, 1000)
	for i := range data {
		data[i] = float64(i)
	}

	have := bs.Bootstrap(data, 10000, trueMean)
	want := trueMean(data)

	haveMean := trueMean(have)
	assert.InEpsilon(t, want, haveMean, 0.001)

	for _, v := range have {
		assert.InEpsilon(t, want, v, 0.1)
	}
}

func trueMean(data []float64) float64 {
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func benchBootstrapMean(b *testing.B, bs Bootstrapper, input []float64) {
	b.Run("1 sample", func(b *testing.B) {
		b.SetBytes(int64(len(input) * 8))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = bs.Mean(input, 1)
		}
	})
	b.Run("1k samples", func(b *testing.B) {
		b.SetBytes(int64(len(input) * 8))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = bs.Mean(input, 1000)
		}
	})
	b.Run("10k samples", func(b *testing.B) {
		b.SetBytes(int64(len(input) * 8))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = bs.Mean(input, 10000)
		}
	})
}

func BenchmarkBootstrapper(b *testing.B) {
	bs := New(rand.NewSource(1).(rand.Source64))
	b.Run("100 input floats", func(b *testing.B) {
		input := make([]float64, 100)
		for i := 0; i < len(input); i++ {
			input[i] = float64(i)
		}
		benchBootstrapMean(b, bs, input)
	})
	b.Run("1k input floats", func(b *testing.B) {
		input := make([]float64, 1000)
		for i := 0; i < len(input); i++ {
			input[i] = float64(i)
		}
		benchBootstrapMean(b, bs, input)
	})
	b.Run("100k input floats", func(b *testing.B) {
		input := make([]float64, 100000)
		for i := 0; i < len(input); i++ {
			input[i] = float64(i)
		}
		benchBootstrapMean(b, bs, input)
	})
}
