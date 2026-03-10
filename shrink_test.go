package pngscaled

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"sync"
	"testing"

	"github.com/kovidgoyal/imaging"
)

type largeResizeCase struct {
	name       string
	srcW, srcH int
	dstW, dstH int
}

var largeResizeCases = []largeResizeCase{
	{name: "3456x2234_to_1920x1080", srcW: 3456, srcH: 2234, dstW: 1920, dstH: 1080},
	{name: "3456x2234_to_32x32", srcW: 3456, srcH: 2234, dstW: 32, dstH: 32},
}

var (
	largePNGCacheMu sync.Mutex
	largePNGCache   = map[string][]byte{}
)

func benchmarkPNGData(tb testing.TB, w, h int) []byte {
	tb.Helper()

	key := fmt.Sprintf("%dx%d", w, h)
	largePNGCacheMu.Lock()
	data, ok := largePNGCache[key]
	largePNGCacheMu.Unlock()
	if ok {
		return data
	}

	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		row := img.Pix[y*img.Stride : y*img.Stride+w*4]
		for x := 0; x < w; x++ {
			i := x * 4
			row[i+0] = uint8((x + y) & 0xff)
			row[i+1] = uint8((x / 3) & 0xff)
			row[i+2] = uint8((y / 5) & 0xff)
			row[i+3] = 0xff
		}
	}

	var buf bytes.Buffer
	enc := png.Encoder{CompressionLevel: png.BestSpeed}
	if err := enc.Encode(&buf, img); err != nil {
		tb.Fatalf("encode benchmark PNG %s: %v", key, err)
	}
	data = buf.Bytes()

	largePNGCacheMu.Lock()
	if cached, exists := largePNGCache[key]; exists {
		data = cached
	} else {
		largePNGCache[key] = data
	}
	largePNGCacheMu.Unlock()

	return data
}

func benchmarkResizeLargePNGScaled(b *testing.B, tc largeResizeCase) {
	data := benchmarkPNGData(b, tc.srcW, tc.srcH)

	b.SetBytes(int64(tc.srcW * tc.srcH * 4))
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if _, err := Decode(bytes.NewReader(data), tc.dstW, tc.dstH, MitchellNetravali); err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkResizeLargeStdlibImaging(b *testing.B, tc largeResizeCase) {
	data := benchmarkPNGData(b, tc.srcW, tc.srcH)

	b.SetBytes(int64(tc.srcW * tc.srcH * 4))
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		decoded, err := png.Decode(bytes.NewReader(data))
		if err != nil {
			b.Fatal(err)
		}
		_ = imaging.Resize(decoded, tc.dstW, tc.dstH, imaging.MitchellNetravali)
	}
}

func BenchmarkResizeLargePNGScaled(b *testing.B) {
	for _, tc := range largeResizeCases {
		tc := tc
		b.Run(tc.name, func(b *testing.B) {
			benchmarkResizeLargePNGScaled(b, tc)
		})
	}
}

func BenchmarkResizeLargeStdlibImaging(b *testing.B) {
	for _, tc := range largeResizeCases {
		tc := tc
		b.Run(tc.name, func(b *testing.B) {
			benchmarkResizeLargeStdlibImaging(b, tc)
		})
	}
}
