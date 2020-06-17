package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"sync"
	"time"

	dgo "github.com/bwmarrin/discordgo"
	"github.com/kavorite/discord-spool"
	pb "github.com/schollz/progressbar/v3"
)

func bytes(src string) (b int, err error) {
	var (
		s float64
		u rune
	)
	istrm := strings.NewReader(src)
	for {
		var n int
		n, err = fmt.Fscanf(istrm, "%f%c", &s, &u)
		unit := 1
		if n > 1 {
			switch u {
			case 'g', 'G':
				unit <<= 10
				fallthrough
			case 'm', 'M':
				unit <<= 10
				fallthrough
			case 'k', 'K':
				unit <<= 10
			default:
				err = fmt.Errorf("unit '%c' not recognized", u)
				return
			}
		}
		b += int(math.Round(s * float64(unit)))
		if err != nil {
			if err == io.EOF {
				err = nil
			}
			return
		}
	}
}

type Semaphore chan struct{}

func (sig Semaphore) Rsrv(n int) {
	for i := 1; i <= n; i++ {
		sig <- struct{}{}
	}
}

func (sig Semaphore) Free(n int) {
	for i := 1; i <= n; i++ {
		<-sig
	}
}

var (
	token    string
	datamass string
	wordpath string
	docsize  uint
)

func main() {
	flag.StringVar(&token, "T", "", "Discord authentication token")
	flag.StringVar(&datamass, "B", "8k", "datamass to retrieve from each channel in units of [K]iB, [M]iB, and [G]iB")
	flag.StringVar(&wordpath, "vocab", "", "path to word2vec embeddings")
	flag.UintVar(&docsize, "doc", 512, "number of lexical units to include in a single content-block")
	flag.Parse()
	if token == "" {
		token = os.Getenv("DMSEARCH_TOKEN")
	}
	if token == "" {
		panic("missing authentication token")
	}
	client, err := dgo.New(token)
	if err != nil {
		panic(err)
	}
	maxbytec, err := bytes(datamass)
	if err != nil {
		panic(err)
	}
	istrm, err := os.Open(wordpath)
	if err != nil {
		panic(err)
	}
	vocab := &Embeddings{}
	fmt.Println("Buffer embeddings...")
	end, err := istrm.Seek(0, os.SEEK_END)
	if err != nil {
		panic(err)
	}
	if _, err = istrm.Seek(0, os.SEEK_SET); err != nil {
		panic(err)
	}
	bar := pb.NewOptions64(end, pb.OptionShowBytes(true))
	go func() {
		var prev, n int64
		for n < end {
			time.Sleep(time.Millisecond * 10)
			if n, err = istrm.Seek(0, os.SEEK_CUR); err != nil {
				panic(err)
			}
			bar.Add64(n - prev)
			prev = n
		}
	}()
	if err = vocab.ReadBin(istrm); err != nil {
		panic(err)
	}
	dms, err := client.UserChannels()
	if err != nil {
		panic(err)
	}
	index := Index{Vocab: vocab}
	indexing := sync.WaitGroup{}
	indexing.Add(len(dms))
	workerlk := make(Semaphore, 32)
	fmt.Println()
	fmt.Println("Index DMs...")
	dmsById := make(map[string]*dgo.Channel, len(dms))
	bar = pb.New(len(dms) * maxbytec)
	for _, dm := range dms {
		dmsById[dm.ID] = dm
		workerlk.Rsrv(1)
		go func(target *dgo.Channel) {
			defer indexing.Done()
			defer workerlk.Free(1)
			bytec := 0
			spool := &spool.T{ChID: target.ID}
			for bytec < maxbytec {
				lens, err := index.Hydrate(client, spool, int(docsize))
				if err != nil {
					if err == io.EOF {
						bar.Add(maxbytec - bytec)
						return
					}
				}
				progress := lens.ContentLength
				bytec += progress
				if bytec >= maxbytec {
					progress -= bytec - maxbytec
				}
				bar.Add(progress)
				if err != nil {
					if err != io.EOF {
						panic(err)
					}
					return
				}
			}
		}(dm)
	}
	indexing.Wait()
	fmt.Println()
	fmt.Printf("Indexing complete.\n> ")
	sc := bufio.NewScanner(os.Stdin)
	for sc.Scan() {
		results := index.QueryBrute(sc.Text())
		k := 8
		if k > len(results) {
			k = len(results)
		}
		results = results[:k]
		fmt.Printf("Found %d hit(s):\n", len(results))
		for _, r := range results {
			recipients := []string{"nobody"}
			if len(dmsById[r.ChID].Recipients) != 0 {
				recipients = make([]string, 0, len(dmsById[r.ChID].Recipients))
				for _, u := range dmsById[r.ChID].Recipients {
					recipients = append(recipients, u.String())
				}
			}
			keyphrases := make([]string, 0, 3)
			for _, phrase := range r.KeyPhrases {
				s := strings.Join(phrase.Tokens, " ")
				s = fmt.Sprintf(`"%s"`, s)
				keyphrases = append(keyphrases, s)
			}
			fmt.Printf("%s; %s: %s\n",
				r.Time.Format("Jan 02 '06 15:04:05"),
				strings.Join(recipients, ", "),
				strings.Join(keyphrases, ", "))
		}
		fmt.Printf("> ")
	}
}
