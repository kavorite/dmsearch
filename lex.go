package main

import (
	"fmt"
	"io"
	"math"
	"strings"
	"unicode"

	"github.com/sajari/fuzzy"
)

func Lex(lexer Lexer, src string) error {
	return LexStrm(lexer, strings.NewReader(src))
}

func LexStrm(lexer Lexer, istrm io.Reader) (err error) {
	t := ""
	_, err = fmt.Fscanf(istrm, "%s ", &t)
	t = lexer.Sanitize(t)
	if err != nil || (t != "" && !lexer.Advance(t)) {
		return
	}
	err = LexStrm(lexer, istrm)
	return
}

type Lexer interface {
	Advance(string) bool
	Sanitizer
}

type Sanitizer interface {
	Sanitize(t string) string
}

type SanitizerFunc func(string) string

func (f SanitizerFunc) Sanitize(t string) string {
	return f(t)
}

var (
	StripPunct = SanitizerFunc(func(t string) string {
		return strings.TrimFunc(t, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsNumber(r)
		})
	})
	ToLower = SanitizerFunc(strings.ToLower)
)

type SanitizerChain []Sanitizer

type SpellChker struct {
	*fuzzy.Model
}

func (cker SpellChker) Sanitize(t string) string {
	return cker.SpellCheck(t)
}

func (chain SanitizerChain) Sanitize(t string) string {
	for _, sanitizer := range chain {
		t = sanitizer.Sanitize(t)
	}
	return t
}

type PassLex struct {
	Sanitizer
}

type CountLex struct {
	n, Max int
	SanitizerChain
}

func (lexer *CountLex) Advance(t string) bool {
	return lexer.n < lexer.Max
}

func (lexer *PassLex) Advance(t string) bool {
	return true
}

type TfIdf struct {
	tf           map[string]int
	df           map[string]int
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

type Doc []string

type DocLexer struct {
	Doc
	Sanitizer
	Cap int
}

func (lexer *DocLexer) Advance(t string) bool {
	t = lexer.Sanitize(t)
	if lexer.Doc == nil {
		lexer.Doc = make(Doc, 0, 1024)
	}
	if lexer.Cap > 0 && len(lexer.Doc) >= lexer.Cap {
		return false
	}
	lexer.Doc = append(lexer.Doc, t)
	return true
}

func (lexer *DocLexer) Finalize() Doc {
	doc := lexer.Doc
	lexer.Doc = nil
	return doc
}

type Spanner struct {
	Lexer
	Span    int
	Context []string
}

func NewSpanner(n int, lexer Lexer) Spanner {
	return Spanner{lexer, n, make([]string, n)}
}

func (spanner *Spanner) Advance(t string) bool {
	spanner.Context = append(spanner.Context, t)
	if len(spanner.Context) > spanner.Span {
		spanner.Context = spanner.Context[1:]
	}
	return spanner.Lexer.Advance(t)
}
