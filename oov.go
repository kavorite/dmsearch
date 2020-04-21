package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
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

func (spanner *CoocSpanner) Advance(t string) (p bool) {
	t = spanner.Sanitize(t)
	p = spanner.Spanner.Advance(t)
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
	return
}

func (spanner *CoocSpanner) PMI() (A *sparse.DOK) {
	n := len(spanner.Vocab)
	A = sparse.NewDOK(n, n)
	for t := range spanner.Dict {
		for w := range spanner.Dict {
			k := fmt.Sprintf("%s/%s", t, w)
			x := spanner.Coocs.Get(k)
			f := spanner.Freqs[t] + spanner.Freqs[w]
			x = math.Log((f - x) / x)
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
			f := spanner.Freqs[t] + spanner.Freqs[w]
			x = math.Log((f - x) / x)
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
	A_hat := mat.NewDense(n, n, make([]float64, n*n))
	rowSums := make([]float64, n)
	A.DoNonZero(func(i, j int, x float64) {
		rowSums[i] += x
	})
	A.DoNonZero(func(i, j int, x float64) {
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
		v.Mul(A, v)
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

type RakeSpanner struct {
	*CoocSpanner
	Vocab
	Stops    BOW
	Phrases  [][]string
	phrase   []string
	Adjoined map[[3]string]float64
	OOV      map[string]*Oneshot
}

func NewRakeSpanner(span int, vocab Vocab, lexer Lexer, stops []string) *RakeSpanner {
	return &RakeSpanner{
		CoocSpanner: NewCoocSpanner(span, 1024, lexer),
		Stops: BOW{
			Dict:      make(map[string]struct{}, 1024),
			Sanitizer: lexer,
		},
		OOV:      make(map[string]*Oneshot, 1024),
		Phrases:  make([][]string, 0, 1024),
		Adjoined: make(map[[3]string]float64, 1024),
		Vocab:    vocab,
	}
}

type Phrase []string

func (phrase Phrase) Append(t string) {
	if phrase == nil {
		phrase = []string{t}
		return
	}
	phrase = append(phrase, t)
}

func (spanner *RakeSpanner) Embed(t string) (v Vec) {
	t = spanner.Sanitize(t)
	v = spanner.Vocab.Embed(t)
	if _, ok := spanner.OOV[t]; v == nil && ok {
		v = spanner.OOV[t].Finalize()
	}
	return
}

func (spanner *RakeSpanner) Advance(t string) (p bool) {
	t = spanner.Sanitize(t)
	p = spanner.CoocSpanner.Advance(t)
	// adjoint phrases
	stops := &spanner.Stops
	for i := 1; i < len(spanner.Context)-1; i++ {
		l := spanner.Context[i-1]
		t := spanner.Context[i]
		w := spanner.Context[i+1]
		if stops.Has(l) || !stops.Has(t) || stops.Has(w) {
			continue
		}
		adjoint := [3]string{l, t, w}
		if _, ok := spanner.Adjoined[adjoint]; !ok {
			spanner.Adjoined[adjoint] = 0
		}
		spanner.Adjoined[adjoint]++
	}
	if len(spanner.Context) > 2 {
		center := len(spanner.Context) / 2
		w := spanner.Context[center]
		if spanner.Vocab.Embed(w) == nil {
			if _, ok := spanner.OOV[w]; !ok {
				spanner.OOV[w] = &Oneshot{}
			}
			ctx := spanner.Context
			for _, l := range append(ctx[:center], ctx[center+1:]...) {
				spanner.OOV[w].Add(spanner.Embed(l))
			}
		}
	}
	w := spanner.Context[0]
	if !stops.Has(w) {
		if spanner.phrase == nil {
			spanner.phrase = make([]string, spanner.Span)
		}
		spanner.phrase = append(spanner.phrase, w)
	} else {
		spanner.phrase = nil
		spanner.Phrases = append(spanner.Phrases, spanner.phrase)
	}
	return
}

type ScoredPhrase struct {
	Phrase
	Score float64
}

func (spanner *RakeSpanner) ScoredPhrases() (scored []*ScoredPhrase) {
	rankVec := spanner.TextRank(1e-3, 0.15)
	rankDict := make(map[string]float64, len(rankVec))
	spmi := spanner.SPMI()
	for i, t := range spanner.CoocSpanner.Vocab {
		rankDict[t] = rankVec[i]
	}
	phrases := spanner.Phrases
	scored = make([]*ScoredPhrase, 0, len(phrases))
	for _, phrase := range phrases {
		if len(phrase) == 0 {
			continue
		}
		t := phrase[0]
		score := rankDict[t]
		if len(phrase) > 1 {
			i := spanner.Dict[t]
			for _, w := range phrase[1:] {
				j := spanner.Dict[w]
				score += spmi.At(i, j)
			}
		}
		if len(phrase) > 1 {
			score /= float64(len(phrase) - 1)
		}
		ent := &ScoredPhrase{phrase, score}
		scored = append(scored, ent)
	}
	for adjoint, frequency := range spanner.Adjoined {
		score := rankDict[adjoint[0]]
		score += rankDict[adjoint[2]]
		score *= math.Log(frequency + 1)
		ent := &ScoredPhrase{adjoint[:], score}
		scored = append(scored, ent)
	}
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})
	return
}
