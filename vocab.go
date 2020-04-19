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

// Sim computes cosine similarity between two Vecs.
func (v Vec) Sim(u Vec) float32 {
	uBlas, vBlas := u.ToBlas(), v.ToBlas()
	return blas32.Dot(uBlas, vBlas) / blas32.Nrm2(uBlas) / blas32.Nrm2(vBlas)
}

func (v Vec) AtVec(i int) float64 {
	return float64(v[i])
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

// Jxt is the juxtaposition between two word vectors. Higher is more similar.
func (v Vec) Jxt(u Vec) float32 {
	uBlas, vBlas := u.ToBlas(), v.ToBlas()
	a, b := blas32.Nrm2(uBlas), blas32.Nrm2(vBlas)
	return blas32.Dot(uBlas, vBlas) / a / b
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
	blas32.Vector
	SampleCount int
}

func (eb *ALaCarte) Advance(t string) (p bool) {
	p = eb.Lexer.Advance(t)
	u := eb.Vocab.Embed(t)
	if u == nil {
		return
	}
	v := u.ToBlas()
	if eb.Vector.Data == nil {
		eb.Vector = make(Vec, eb.Dim()).ToBlas()
	}
	blas32.Axpy(1, v, eb.Vector)
	eb.SampleCount++
	return
}

func (eb *ALaCarte) Finalize() (x Vec) {
	if eb.Vector.Data == nil {
		eb.Vector = make(Vec, eb.Dim()).ToBlas()
	}
	blas32.Scal(1/float32(eb.SampleCount), eb.Vector)
	// v := blas32.General{
	// 	Rows: eb.Dim(), Cols: 1, // [1×n]
	// 	Stride: 1, Data: eb.Vector.Data,
	// }
	A := inductionMatrix // [n×n]
	// v = vA
	// blas32.Gemm(blas.NoTrans, blas.NoTrans, 1, v, A, 0, v)
	// x = Vec(v.Data)
	v := mat.NewDense(eb.Dim(), 1, nil)
	v.Mul(A, Vec(eb.Vector.Data))
	x = make(Vec, eb.Dim())
	for i := range x {
		x[i] = float32(v.RawMatrix().Data[i])
	}
	eb.Vector = make(Vec, eb.Dim()).ToBlas()
	eb.SampleCount = 0
	return
}
