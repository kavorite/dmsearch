package main

import (
	"sort"
	"sync"
	"time"

	"github.com/Bithack/go-hnsw"
	dgo "github.com/bwmarrin/discordgo"
	"github.com/kavorite/discord-snowflake"
	"github.com/kavorite/discord-spool"
)

type Prism struct {
	*spool.T
	Vocab
	Width int
}

type Lens struct {
	time.Time
	Vec
	KeyPhrases    []*ScoredPhrase
	ChID          string
	ContentLength int
}

func (spl *Prism) Slide(s *dgo.Session) (distillation *Lens, err error) {
	eb := ALaCarte{
		Lexer: &CountLex{
			Max:            spl.Width,
			SanitizerChain: SanitizerChain{StripPunct, ToLower},
		},
		Vocab: spl.Vocab,
	}
	bytec := 0
	spanner := NewRakeSpanner(5, &eb, &eb, nltkStops)
	err = spl.Unroll(s, func(msg *dgo.Message) bool {
		Lex(spanner, msg.Content)
		bytec += len([]byte(msg.Content))
		msgid, _ := snowflake.Parse(msg.ID)
		if eb.SampleCount >= spl.Width {
			distillation = &Lens{
				Time:          msgid.Time(),
				ContentLength: bytec,
				Vec:           eb.Finalize(),
				ChID:          msg.ChannelID,
				KeyPhrases:    spanner.ScoredPhrases(),
			}
			return false
		}
		return true
	})
	return
}

type Index struct {
	Vocab
	cluster *hnsw.Hnsw
	qledger map[uint32]*Lens
	ledgerc int
	sync.RWMutex
}

func (index *Index) Hydrate(client *dgo.Session, spl *spool.T, width int) (lens *Lens, err error) {
	prism := Prism{spl, index.Vocab, width}
	lens, err = prism.Slide(client)
	if err != nil {
		return
	}
	if index.cluster == nil {
		m, efConstruction, zero := 32, 256, make(hnsw.Point, index.Dim())
		index.ledgerc = 1024
		index.cluster = hnsw.New(m, efConstruction, zero)
		index.Lock()
		index.qledger = make(map[uint32]*Lens, index.ledgerc)
		index.Unlock()
		index.cluster.Grow(index.ledgerc)
	} else if len(index.qledger) >= int(0.8*float64(index.ledgerc)) {
		index.ledgerc *= 2
		index.cluster.Grow(index.ledgerc)
	}
	index.RLock()
	id := uint32(len(index.qledger) + 1)
	index.RUnlock()
	index.Lock()
	index.qledger[id] = lens
	index.Unlock()
	index.cluster.Add(hnsw.Point(lens.Vec), id)
	return
}

type Result struct {
	*Lens
	Distance float32
}

func (index *Index) Query(q string) (results []Result) {
	eb := ALaCarte{
		Vocab: index.Vocab,
		Lexer: &PassLex{SanitizerChain{StripPunct, ToLower}},
	}
	Lex(&eb, q)
	v := hnsw.Point(eb.Finalize())
	items := index.cluster.Search(v, 64, len(index.qledger)).Items()
	// items := index.cluster.SearchBrute(v, len(index.qledger)).Items()
	results = make([]Result, len(items))
	for i, item := range items {
		results[i] = Result{index.qledger[item.ID], item.D}
	}
	return
}

func (index *Index) QueryBrute(q string) (results []Result) {
	eb := ALaCarte{
		Vocab: index.Vocab,
		Lexer: &PassLex{SanitizerChain{StripPunct, ToLower}},
	}
	Lex(&eb, q)
	v := eb.Finalize()
	results = make([]Result, 0, len(index.qledger))
	for _, lens := range index.qledger {
		results = append(results, Result{lens, v.Sim(lens.Vec)})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance > results[j].Distance
	})
	return
}
