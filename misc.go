package main

import (
	"math"

	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/mat"
)

type TfIdf struct {
	tf, df       map[string]int
	touch        map[string]struct{}
	docs, docLen int
}

func (counter *TfIdf) Init() {
	counter.tf = make(map[string]int, 1024)
	counter.df = make(map[string]int, 1024)
	counter.touch = make(map[string]struct{}, 512)
}

func (counter *TfIdf) Inc(t string) {
	if counter.tf == nil {
		counter.Init()
	}
	counter.tf[t] += 1
	counter.touch[t] = struct{}{}
}

func (counter *TfIdf) Doc() {
	if counter.tf == nil {
		counter.Init()
	}
	counter.docs += 1
	for t := range counter.touch {
		counter.df[t] += 1
	}
	if len(counter.touch) > counter.docLen {
		counter.docLen = len(counter.touch)
	}
	counter.touch = make(map[string]struct{}, 2*counter.docLen)
}

func (counter *TfIdf) Oneshot(t string) float64 {
	if counter.docs == 0 {
		counter.Doc()
	}
	if _, ok := counter.tf[t]; !ok {
		return 0
	}
	tf := float64(counter.tf[t])
	df := float64(counter.df[t])
	d := float64(counter.docs)
	return math.Log(tf+1) * math.Log((d-df)/df)
}

func (counter *TfIdf) Collate() map[string]float64 {
	final := make(map[string]float64, len(counter.tf))
	for t := range counter.tf {
		final[t] = counter.Oneshot(t)
	}
	return final
}

type Counter map[string]float64

func (ctr Counter) Get(k string) float64 {
	if x, ok := ctr[k]; ok {
		return x
	}
	return 0
}

func (ctr Counter) Inc(k string, x float64) {
	ctr[k] = ctr.Get(k) + x
}

type BOW struct {
	Dict map[string]struct{}
	Sanitizer
}

func (bag *BOW) Advance(t string) bool {
	if bag.Dict == nil {
		bag.Dict = make(map[string]struct{}, 1024)
	}
	bag.Ins(t)
	return true
}

func (bag *BOW) Ins(t string) {
	bag.Dict[bag.Sanitize(t)] = struct{}{}
}

func (bag *BOW) Has(t string) bool {
	_, ok := bag.Dict[bag.Sanitize(t)]
	return ok
}

func (bag *BOW) Del(t string) {
	delete(bag.Dict, bag.Sanitize(t))
}

type Oneshot struct {
	Vec
	SampleCount int
}

func (cma *Oneshot) Add(v Vec) {
	if v == nil {
		return
	}
	if cma.Vec == nil {
		cma.Vec = make(Vec, v.Dim())
		copy(cma.Vec, v)
		return
	}
	u := cma.ToBlas()
	blas32.Axpy(1, u, v.ToBlas())
	cma.SampleCount++
	return
}

func (cma *Oneshot) Finalize() (u Vec) {
	if cma.Vec == nil {
		return
	}
	u = make(Vec, cma.Dim())
	blas32.Scal(1/float32(cma.SampleCount), u.ToBlas())
	copy(u, cma.Vec)
	v := mat.NewDense(cma.Dim(), 1, nil)
	A := inductionMatrix
	v.Mul(A, u)
	for i := range u {
		u[i] = float32(v.RawMatrix().Data[i])
	}
	return
}
