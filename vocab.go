package main

import (
	"encoding/binary"
	"fmt"
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/mat"
	"io"
	"sync"
)

// Vec represents a lexical embedding.
type Vec []float32

// Sim is the cosine similarity between two word vectors. Higher is more
// similar.
func (v Vec) Sim(u Vec) float32 {
	if len(u) != len(v) {
		return 0
	}
	uBlas, vBlas := u.ToBlas(), v.ToBlas()
	a, b := blas32.Nrm2(uBlas), blas32.Nrm2(vBlas)
	return blas32.Dot(uBlas, vBlas) / a / b
}

func (v Vec) Scale(alpha float32) Vec {
	u := v.ToBlas()
	blas32.Scal(alpha, u)
	return v
}

func (v Vec) AtVec(i int) float64 {
	return float64(v[i])
}

func (v Vec) Dim() int {
	return len(v)
}

func (v Vec) Dims() (m, n int) {
	m, n = len(v), 1
	return
}

func (v Vec) At(i, j int) float64 {
	if j != 0 {
		panic("j ≠ 0")
	}
	return float64(v[i])
}

func (v Vec) T() mat.Matrix {
	return mat.Transpose{v}
}

func (v Vec) ToBlas() blas32.Vector {
	return blas32.Vector{len(v), 1, v}
}

type Vocab interface {
	Len() int
	Dim() int
	Embed(string) Vec
}

type Embeddings struct {
	Dict map[string]Vec
	sync.RWMutex
	dim int
}

func (eb *Embeddings) Len() int {
	eb.RLock()
	defer eb.RUnlock()
	return len(eb.Dict)
}

func (eb *Embeddings) Dim() int {
	return eb.dim
}

func (eb *Embeddings) OOV(t string) bool {
	eb.RLock()
	defer eb.RUnlock()
	_, ok := eb.Dict[t]
	return ok
}

func (eb *Embeddings) Embed(t string) Vec {
	eb.RLock()
	defer eb.RUnlock()
	if v, ok := eb.Dict[t]; ok {
		return v
	}
	return nil
}

func (eb *Embeddings) ReadBin(istrm io.Reader) (err error) {
	var (
		wordc, dimen int
		t            string
	)
	_, err = fmt.Fscanf(istrm, "%d %d\n", &wordc, &dimen)
	if err != nil {
		return
	}
	eb.Dict = make(map[string]Vec, wordc)
	eb.dim = dimen
	for i := 0; i < wordc; i++ {
		_, err = fmt.Fscanf(istrm, "%s", &t)
		if err != nil {
			if err == io.EOF {
				err = nil
			}
			return
		}
		embedding := make(Vec, dimen)
		if err = binary.Read(istrm, binary.LittleEndian, embedding); err != nil {
			return
		}
		eb.Dict[t] = embedding
	}
	return
}

func (eb *Embeddings) EmbedTxt(istrm io.Reader) (err error) {
	var (
		wordc, dimen int
		t            string
	)
	_, err = fmt.Fscanf(istrm, "%d %d", &wordc, &dimen)
	if err != nil {
		return
	}
	eb.Dict = make(map[string]Vec, wordc)
	eb.dim = dimen
	for i := 0; i < wordc; i++ {
		_, err = fmt.Fscanf(istrm, "%s", &t)
		if err != nil {
			if err == io.EOF {
				err = nil
			}
			return
		}
		embedding := make([]float32, dimen)
		for b := 0; b < dimen; b++ {
			_, err = fmt.Fscanf(istrm, "%f", &embedding[b])
			if err != nil {
				return
			}
		}
	}
	return
}

type ALaCarte struct {
	Vocab
	Lexer
	Oneshot
}

func (eb *ALaCarte) Advance(t string) (p bool) {
	if !eb.Lexer.Advance(t) {
		return false
	}
	eb.Oneshot.Add(eb.Embed(t))
	return true
}

func (eb *ALaCarte) Finalize() (x Vec) {
	x = eb.Oneshot.Finalize()
	eb.Oneshot = Oneshot{}
	return
}
