package main

import (
	"fmt"
	"io"
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

type Spanner struct {
	Lexer
	Span    int
	Context []string
}

func NewSpanner(span int, lexer Lexer) *Spanner {
	return &Spanner{lexer, span, make([]string, 0, span+1)}
}

func (spanner *Spanner) Advance(t string) bool {
	spanner.Context = append(spanner.Context, spanner.Sanitize(t))
	if len(spanner.Context) > spanner.Span {
		spanner.Context = spanner.Context[1:]
	}
	return spanner.Lexer.Advance(t)
}
