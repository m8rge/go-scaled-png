// shrink-samples decodes PNG files and writes scaled-down versions to an
// output directory for visual inspection.
//
// Usage:
//
//	go run ./cmd/shrink-samples [flags] [file ...]
//
// If no files are given, all *.png files in testdata/pngsuite/ are processed.
// Flags:
//
//	-out dir   output directory (default "out")
//	-w  int    target width  in pixels (default 64)
//	-h  int    target height in pixels (default 64)
package main

import (
	"flag"
	"fmt"
	"image/png"
	"os"
	"path/filepath"

	pngscaled "github.com/m8rge/go-scaled-png"
)

func main() {
	outDir := flag.String("out", "out", "output directory")
	w := flag.Int("w", 16, "target width")
	h := flag.Int("h", 16, "target height")
	flag.Parse()

	inputs := flag.Args()
	if len(inputs) == 0 {
		matches, err := filepath.Glob("testdata/pngsuite/*.png")
		if err != nil || len(matches) == 0 {
			fmt.Fprintln(os.Stderr, "no input files found in testdata/pngsuite/")
			os.Exit(1)
		}
		inputs = matches
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "mkdir %s: %v\n", *outDir, err)
		os.Exit(1)
	}

	ok, fail := 0, 0
	for _, path := range inputs {
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "open %s: %v\n", path, err)
			fail++
			continue
		}
		decoded, err := pngscaled.Decode(f, *w, *h, pngscaled.MitchellNetravali)
		f.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "decode %s: %v\n", path, err)
			fail++
			continue
		}

		outPath := filepath.Join(*outDir, filepath.Base(path))
		out, err := os.Create(outPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "create %s: %v\n", outPath, err)
			fail++
			continue
		}
		encErr := png.Encode(out, decoded)
		out.Close()
		if encErr != nil {
			fmt.Fprintf(os.Stderr, "encode %s: %v\n", outPath, encErr)
			fail++
			continue
		}
		fmt.Printf("%-40s -> %s  %v\n", path, outPath, decoded.Bounds())
		ok++
	}

	fmt.Printf("\n%d ok, %d failed\n", ok, fail)
	if fail > 0 {
		os.Exit(1)
	}
}
