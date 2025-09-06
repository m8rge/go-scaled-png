package pngscaled

import (
	"bytes"
	"image"
	"image/color"
	stdpng "image/png"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

// helper encodes an image to standard PNG bytes (non-interlaced 8-bit)
func encodePNG(t *testing.T, img image.Image) []byte {
	t.Helper()
	var buf bytes.Buffer
	enc := stdpng.Encoder{CompressionLevel: stdpng.DefaultCompression}
	if err := enc.Encode(&buf, img); err != nil {
		t.Fatalf("png encode: %v", err)
	}
	return buf.Bytes()
}

// minimalInterlacedPNG returns a tiny PNG byte slice whose IHDR sets interlace=1 (Adam7).
// We don't need valid IDAT for our decoder to reject it: the decoder errors immediately
// after reading IHDR if interlace != 0.
func minimalInterlacedPNG() []byte {
	// PNG sig
	b := []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}
	// IHDR length (13)
	b = append(b, 0, 0, 0, 13)
	// IHDR type
	b = append(b, 'I', 'H', 'D', 'R')
	// IHDR payload: w=1,h=1,bitDepth=8,colorType=6,compression=0,filter=0,interlace=1
	b = append(b,
		0, 0, 0, 1, // width
		0, 0, 0, 1, // height
		8, // bit depth
		6, // color type RGBA
		0, // compression
		0, // filter method
		1, // interlace = Adam7
	)
	// fake CRC (ignored by our MVP)
	b = append(b, 0, 0, 0, 0)
	// IEND chunk (length=0, type IEND, fake CRC)
	b = append(b, 0, 0, 0, 0, 'I', 'E', 'N', 'D', 0, 0, 0, 0)
	return b
}

func TestDecodeBasicPNG(t *testing.T) {
	// Generate a 1x1 red PNG with the stdlib encoder (valid IDAT/zlib)
	img1x1 := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img1x1.SetRGBA(0, 0, color.RGBA{255, 0, 0, 255})
	data := encodePNG(t, img1x1)

	img, err := Decode(bytes.NewReader(data), nil)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if img.Bounds().Dx() != 1 || img.Bounds().Dy() != 1 {
		t.Fatalf("unexpected image size: %v", img.Bounds())
	}
	r, g, b, a := img.At(0, 0).RGBA()
	if r>>8 != 255 || g>>8 != 0 || b>>8 != 0 || a>>8 != 255 {
		t.Fatalf("expected red pixel, got RGBA=(%d,%d,%d,%d)", r>>8, g>>8, b>>8, a>>8)
	}
}

func TestLineCallback(t *testing.T) {
	img1x1 := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img1x1.SetRGBA(0, 0, color.RGBA{255, 0, 0, 255})
	data := encodePNG(t, img1x1)

	called := false
	_, err := Decode(bytes.NewReader(data), func(y int, row []byte) {
		called = true
		if len(row) != 4 {
			t.Fatalf("expected 4 bytes per row, got %d", len(row))
		}
	})
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if !called {
		t.Fatal("callback was not called")
	}
}

func TestDecode_RGBA_Roundtrip(t *testing.T) {
	w, h := 7, 5
	src := image.NewRGBA(image.Rect(0, 0, w, h))
	// Fill with a pattern that exercises all filters
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r := uint8((x*37 + y*17) & 0xFF)
			g := uint8((x*19 + y*53) & 0xFF)
			b := uint8((x*11 + y*29) & 0xFF)
			a := uint8(255)
			src.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: a})
		}
	}
	data := encodePNG(t, src)

	// Track callback rows for correctness
	seenRows := 0
	got, err := Decode(bytes.NewReader(data), func(y int, row []byte) {
		if len(row) != w*4 {
			t.Fatalf("row len=%d want %d", len(row), w*4)
		}
		seenRows++
	})
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if seenRows != h {
		t.Fatalf("callback rows=%d want %d", seenRows, h)
	}

	// Pixel-by-pixel compare
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g := got.RGBAAt(x, y)
			s := src.RGBAAt(x, y)
			if g != s {
				t.Fatalf("pixel mismatch at (%d,%d): got=%v want=%v", x, y, g, s)
			}
		}
	}
}

func TestDecode_Gray(t *testing.T) {
	w, h := 4, 3
	src := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			src.SetGray(x, y, color.Gray{Y: uint8(x*17 + y*13)})
		}
	}
	data := encodePNG(t, src)

	got, err := Decode(bytes.NewReader(data), nil)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g := got.RGBAAt(x, y)
			yv := uint8(x*17 + y*13)
			want := color.RGBA{yv, yv, yv, 255}
			if g != want {
				t.Fatalf("pixel mismatch at (%d,%d): got=%v want=%v", x, y, g, want)
			}
		}
	}
}

func TestDecode_PalettedWithTRNS(t *testing.T) {
	w, h := 3, 2
	// Force 8-bit indexed by using a 256-entry palette.
	pal := make(color.Palette, 256)
	pal[0] = color.NRGBA{0x10, 0x20, 0x30, 0xFF} // index 0
	pal[1] = color.NRGBA{0xAA, 0xBB, 0xCC, 0x00} // index 1 transparent via tRNS
	pal[2] = color.NRGBA{0x00, 0xFF, 0x00, 0xFF} // index 2
	for i := 3; i < 256; i++ {                   // fill remaining with opaque greyscale
		v := uint8(i)
		pal[i] = color.NRGBA{v, v, v, 0xFF}
	}
	src := image.NewPaletted(image.Rect(0, 0, w, h), pal)
	// Pattern uses index 1 (transparent) in a few spots
	copy(src.Pix, []uint8{0, 1, 2, 2, 1, 0})

	data := encodePNG(t, src)

	got, err := Decode(bytes.NewReader(data), nil)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	// Validate pixels (index 1 must produce A=0)
	want := []color.RGBA{
		{0x10, 0x20, 0x30, 0xFF}, {0xAA, 0xBB, 0xCC, 0x00}, {0x00, 0xFF, 0x00, 0xFF},
		{0x00, 0xFF, 0x00, 0xFF}, {0xAA, 0xBB, 0xCC, 0x00}, {0x10, 0x20, 0x30, 0xFF},
	}
	i := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g := got.RGBAAt(x, y)
			if g != want[i] {
				t.Fatalf("pixel %d (%d,%d): got=%v want=%v", i, x, y, g, want[i])
			}
			i++
		}
	}
}

func TestDecode_UnsupportedInterlaceAndBitDepth(t *testing.T) {
	// Interlaced should error: craft a minimal PNG with IHDR.interlace=1
	{
		data := minimalInterlacedPNG()
		if _, err := Decode(bytes.NewReader(data), nil); err == nil {
			t.Fatalf("expected error for interlaced PNG")
		}
	}

	// 16-bit depth should error (encode RGBA64)
	{
		w, h := 2, 2
		img := image.NewRGBA64(image.Rect(0, 0, w, h))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				img.SetRGBA64(x, y, color.RGBA64{R: 0xFFFF, G: 0, B: 0, A: 0xFFFF})
			}
		}
		var buf bytes.Buffer
		if err := stdpng.Encode(&buf, img); err != nil {
			t.Fatalf("encode 16-bit: %v", err)
		}
		if _, err := Decode(bytes.NewReader(buf.Bytes()), nil); err == nil {
			t.Fatalf("expected error for 16-bit PNG")
		}
	}
}

func BenchmarkDecode_RGBOpaque(b *testing.B) {
	w, h := 256, 256
	src := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			src.SetRGBA(x, y, color.RGBA{uint8(x), uint8(y), uint8(x ^ y), 0xFF})
		}
	}
	data := encodePNG(&testing.T{}, src)

	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Decode(bytes.NewReader(data), nil); err != nil {
			b.Fatal(err)
		}
	}
}

func TestRealImage(t *testing.T) {
	file, err := os.Open("2.png")
	if err != nil {
		t.Fatalf("open failed: %v", err)
	}
	img, err := Decode(file, nil)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	out, err := os.OpenFile("2-out.png", os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		t.Fatalf("open failed: %v", err)
	}
	err = stdpng.Encode(out, img)
	require.NoError(t, err)
}
