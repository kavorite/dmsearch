package main

import (
	"github.com/jdkato/prose/v2"
	"sort"

	snowball "github.com/kljensen/snowball/english"
)

// NLTK-generated list of stop-words
var nltkStops = []string{
	"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
	"you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
	"yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers",
	"herself", "it", "it's", "its", "itself", "they", "them", "their",
	"theirs", "themselves", "what", "which", "who", "whom", "this", "that",
	"that'll", "these", "those", "am", "is", "are", "was", "were", "be",
	"been", "being", "have", "has", "had", "having", "do", "does", "did",
	"doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
	"until", "while", "of", "at", "by", "for", "with", "about", "against",
	"between", "into", "through", "during", "before", "after", "above",
	"below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
	"under", "again", "further", "then", "once", "here", "there", "when",
	"where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
	"other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
	"than", "too", "very", "s", "t", "can", "will", "just", "don", "don't",
	"should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain",
	"aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn",
	"doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn",
	"isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn",
	"needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't",
	"weren", "weren't", "won", "won't", "wouldn", "wouldn't",
}

type RAKE struct {
	*CoocSpanner
	Stops       BOW
	Phrases     [][]string
	phrases     [][]string
	phrase      []string
	unstem      []string
	initialized bool
	ngc         int
}

func (tr *RAKE) Doc() {
	if len(tr.unstem) == 0 {
		return
	}
	tr.phrase = make([]string, 0, 8)
	tr.unstem = make([]string, 0, 8)
}

func (tr *RAKE) Advance(t string) bool {
	t = tr.CoocSpanner.Sanitize(t)
	if t == "" {
		return true
	}
	if !tr.CoocSpanner.Advance(t) {
		return false
	}
	if len(tr.phrase) == tr.ngc || tr.Stops.Has(t) {
		tr.Doc()
		return true
	}
	st := snowball.Stem(normalize(t), false)
	if st == "" {
		return true
	}
	if len(tr.phrase) > 0 {
		tr.phrase = append(tr.phrase, st)
		tr.unstem = append(tr.unstem, t)
	}
	if len(tr.Context) < 3 {
		return true
	}
	// add adjoint phrases
	l, t, r := tr.Context[0], tr.Context[1], tr.Context[2]
	if tr.Stops.Has(l) || tr.Stops.Has(r) {
		return true
	}
	tr.Phrases = append(tr.Phrases, []string{l, t, r})
	lst := snowball.Stem(l, false)
	tst := snowball.Stem(t, false)
	rst := snowball.Stem(r, false)
	tr.phrases = append(tr.phrases, []string{lst, tst, rst})
	return true
}

func (tr *RAKE) Init(ngc, size int) {
	if tr.initialized {
		return
	}
	if size < 1 {
		size = 1024
	}
	if ngc < 2 {
		ngc = 5
	}
	tr.ngc = ngc
	tr.CoocSpanner = NewCoocSpanner(ngc, size,
		&PassLex{SanitizerChain{StripPunct, ToLower}})
	tr.Phrases = make([][]string, 0, size)
	tr.phrases = make([][]string, 0, size)
	stops := make([]string, len(nltkStops))
	for i, t := range nltkStops {
		stops[i] = snowball.Stem(t, false)
	}
	tr.Stops = Bag(append(stops, nltkStops...)...)
}

func (tr *RAKE) Ingest(ngc int, src string) {
	tr.Init(ngc, 1024)
	// break on candidate phrases
	// Lex(tr, src)
	doc, _ := prose.NewDocument(src,
		prose.WithTagging(false),
		prose.WithExtraction(false))
	sents := doc.Sentences()
	if len(sents) > 1 {
		for _, sent := range sents {
			tr.Ingest(ngc, sent.Text)
		}
	} else {
		for _, t := range doc.Tokens() {
			tr.Advance(t.Text)
		}
	}
}

type ScoredPhrase struct {
	Weight float64
	Tokens []string
}

func (tr *RAKE) Finalize() (tokens, phrases []ScoredPhrase) {
	A := tr.CoocSpanner.PMI()
	m, n := A.Dims()
	if m != n || n == 0 {
		return
	}
	A_hat := A.ToDense()
	R := make([]float64, n)
	for i := range R {
		for j := range R {
			if j == i {
				continue
			}
			R[i] += A_hat.At(i, j)
		}
	}
	// R := tr.CoocSpanner.TextRank(1e-3, 0.5)
	// n := len(R)
	tokens = make([]ScoredPhrase, n)
	phrases = make([]ScoredPhrase, len(tr.Phrases))
	for i := range tr.Phrases {
		score := float64(0)
		for _, stem := range tr.phrases[i] {
			score += R[tr.Dict[stem]]
		}
		score /= float64(len(tr.Phrases[i]))
		phrases[i] = ScoredPhrase{score, tr.Phrases[i]}
	}
	for t, i := range tr.Dict {
		tokens[i] = ScoredPhrase{R[i], []string{t}}
	}
	sort.Slice(tokens, func(i, j int) bool {
		return tokens[i].Weight > tokens[j].Weight
	})
	sort.Slice(phrases, func(i, j int) bool {
		return phrases[i].Weight > phrases[j].Weight
	})
	return
}
