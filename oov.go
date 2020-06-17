package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
)

func init() {
	rand.Seed(time.Now().Unix())
}

type CoocSpanner struct {
	*Spanner
	Vocab               []string
	Dict                map[string]int
	Freqs, Coocs, Succs Counter
}

func NewCoocSpanner(span, vocab int, lexer Lexer) *CoocSpanner {
	return &CoocSpanner{
		Spanner: NewSpanner(span, lexer),
		Vocab:   make([]string, 0, vocab),
		Dict:    make(map[string]int, vocab),
		Freqs:   make(Counter, vocab),
		Coocs:   make(Counter, vocab),
		Succs:   make(Counter, vocab),
	}
}

func (spanner *CoocSpanner) Advance(t string) bool {
	t = spanner.Sanitize(t)
	if !spanner.Spanner.Advance(t) {
		return false
	}
	if _, ok := spanner.Dict[t]; !ok {
		spanner.Dict[t] = len(spanner.Dict)
		spanner.Vocab = append(spanner.Vocab, t)
	}
	for i, t := range spanner.Context {
		spanner.Freqs.Inc(t, 1/float64(spanner.Span))
		for j, w := range spanner.Context {
			k := fmt.Sprintf("%s/%s", t, w)
			spanner.Coocs.Inc(k, 1/float64(spanner.Span))
			if j > i {
				spanner.Succs.Inc(k, 1/float64(spanner.Span-1))
			}
		}
	}
	return true
}

func (spanner *CoocSpanner) PMI() (A *sparse.DOK) {
	n := len(spanner.Vocab)
	A = sparse.NewDOK(n, n)
	for t := range spanner.Dict {
		for w := range spanner.Dict {
			k := fmt.Sprintf("%s/%s", t, w)
			x := spanner.Coocs.Get(k)
			if x == 0 {
				continue
			}
			f := spanner.Freqs[t] + spanner.Freqs[w]
			x /= f
			i, j := spanner.Dict[t], spanner.Dict[w]
			A.Set(i, j, x)
		}
	}
	return
}

func (spanner *CoocSpanner) SPMI() (A *sparse.DOK) {
	n := len(spanner.Vocab)
	A = sparse.NewDOK(n, n)
	for t := range spanner.Dict {
		for w := range spanner.Dict {
			k := fmt.Sprintf("%s/%s", t, w)
			x := spanner.Succs.Get(k)
			if x == 0 {
				continue
			}
			f := spanner.Freqs[t] + spanner.Freqs[w]
			x /= f
			i, j := spanner.Dict[t], spanner.Dict[w]
			A.Set(i, j, x)
		}
	}
	return
}

func (spanner *CoocSpanner) TextRank(e, d float64) []float64 {
	if d < 0 || d > 1 {
		d = 0.15
	}
	if e < 0 || e > 1 {
		e = 1e-3
	}
	A := spanner.PMI()
	n, _ := A.Dims()
	rowSums := make([]float64, n)
	A.DoNonZero(func(i, j int, x float64) {
		rowSums[i] += x
	})
	A_hat := mat.NewDense(n, n, make([]float64, n*n))
	A.DoNonZero(func(i, j int, x float64) {
		if rowSums[i] == 0 {
			return
		}
		A_hat.Set(i, j, x/rowSums[i])
	})
	A_hat.Apply(func(i, j int, x float64) float64 {
		x *= (1 - d)
		x += d / float64(n)
		return x
	}, A_hat)
	v := mat.NewDense(n, 1, make([]float64, n))
	v.Apply(func(i, j int, x float64) float64 {
		return rand.Float64() / float64(n)
	}, v)
	for {
		u := mat.DenseCopyOf(v)
		v.Mul(A_hat, v)
		u.Sub(u, v)
		u.MulElem(u, u)
		qerr := mat.Sum(u)
		if qerr < e {
			break
		}
	}
	v.Scale(1/mat.Sum(v), v)
	return mat.VecDenseCopyOf(v.ColView(0)).RawVector().Data
}
