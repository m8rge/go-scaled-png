package pngscaled

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestShrinkPNGSuiteTypes verifies that shrinking works for all new color types
// (16-bit and paletted) without panics, and returns the correct Go image type.
// Files are from the public-domain PNG test suite shipped in testdata/pngsuite/.
func TestShrinkPNGSuiteTypes(t *testing.T) {
	// All basn/ft files below are 32×32; we shrink to 16×16.
	cases := []struct {
		file     string
		wantType string // expected fmt.Sprintf("%T", img)
	}{
		{"basn0g16", "*image.Gray16"},    // cbG16, no tRNS  → Gray16
		{"ftbwn0g16", "*image.NRGBA64"}, // cbG16 + tRNS    → NRGBA64
		{"basn4a16", "*image.NRGBA64"},  // cbGA16           → NRGBA64
		{"basn2c16", "*image.RGBA64"},   // cbTC16, no tRNS  → RGBA64
		{"ftbbn2c16", "*image.NRGBA64"}, // cbTC16 + tRNS    → NRGBA64
		{"basn6a16", "*image.NRGBA64"},  // cbTCA16          → NRGBA64
		{"basn3p01", "*image.NRGBA"},    // cbP1             → NRGBA (shrink path)
		{"basn3p02", "*image.NRGBA"},    // cbP2             → NRGBA
		{"basn3p04", "*image.NRGBA"},    // cbP4             → NRGBA
		{"basn3p08", "*image.NRGBA"},    // cbP8             → NRGBA
		{"basn3p08-trns", "*image.NRGBA"}, // cbP8 + tRNS   → NRGBA
	}
	for _, tc := range cases {
		t.Run(tc.file, func(t *testing.T) {
			f, err := os.Open("testdata/pngsuite/" + tc.file + ".png")
			require.NoError(t, err)
			defer f.Close()

			img, err := Decode(f, 16, 16, MitchellNetravali)
			require.NoError(t, err)
			require.Equal(t, image.Rect(0, 0, 16, 16), img.Bounds(),
				"unexpected bounds for %s", tc.file)
			require.Equal(t, tc.wantType, fmt.Sprintf("%T", img),
				"unexpected image type for %s", tc.file)
		})
	}
}

// TestShrinkFlatColor checks pixel correctness: a uniform-color image resampled
// to different dimensions must return the same color within Q15 rounding
// tolerance (≤2 LSBs for 16-bit, ≤1 LSB for 8-bit).
func TestShrinkFlatColor(t *testing.T) {
	const srcW, srcH = 64, 64
	const dstW, dstH = 30, 25

	t.Run("Gray16", func(t *testing.T) {
		const wantY = uint16(0x8642)
		src := image.NewGray16(image.Rect(0, 0, srcW, srcH))
		for y := 0; y < srcH; y++ {
			for x := 0; x < srcW; x++ {
				src.SetGray16(x, y, color.Gray16{Y: wantY})
			}
		}
		img := mustShrink(t, src, dstW, dstH)
		g16 := img.(*image.Gray16)
		for y := 0; y < dstH; y++ {
			for x := 0; x < dstW; x++ {
				got := g16.Gray16At(x, y).Y
				if d := absDiffU16(got, wantY); d > 2 {
					t.Fatalf("at (%d,%d): got %d, want %d (diff=%d)", x, y, got, wantY, d)
				}
			}
		}
	})

	// Go's png.Encode optimizes NRGBA64 with all A=0xffff to cbTC16 (no alpha channel).
	// That path returns *image.RGBA64 from our decoder.
	t.Run("RGBA64_TC16_fullalpha", func(t *testing.T) {
		want := color.NRGBA64{R: 0x3456, G: 0x789a, B: 0xbcde, A: 0xffff}
		src := image.NewNRGBA64(image.Rect(0, 0, srcW, srcH))
		for y := 0; y < srcH; y++ {
			for x := 0; x < srcW; x++ {
				src.SetNRGBA64(x, y, want)
			}
		}
		img := mustShrink(t, src, dstW, dstH)
		r64 := img.(*image.RGBA64) // cbTC16 → RGBA64
		for y := 0; y < dstH; y++ {
			for x := 0; x < dstW; x++ {
				got := r64.RGBA64At(x, y)
				if absDiffU16(got.R, want.R) > 2 || absDiffU16(got.G, want.G) > 2 || absDiffU16(got.B, want.B) > 2 {
					t.Fatalf("at (%d,%d): got %v, want %v", x, y, got, want)
				}
				if got.A != 0xffff {
					t.Fatalf("at (%d,%d) A: got %d, want 0xffff", x, y, got.A)
				}
			}
		}
	})

	// NRGBA64 with A < 0xffff is encoded as cbTCA16 (colortype=6).
	// Returns *image.NRGBA64. Exercises premultiplied-alpha accumulation.
	t.Run("NRGBA64_TCA16_semitransparent", func(t *testing.T) {
		want := color.NRGBA64{R: 0x3456, G: 0x789a, B: 0xbcde, A: 0x8000}
		src := image.NewNRGBA64(image.Rect(0, 0, srcW, srcH))
		for y := 0; y < srcH; y++ {
			for x := 0; x < srcW; x++ {
				src.SetNRGBA64(x, y, want)
			}
		}
		img := mustShrink(t, src, dstW, dstH)
		n64 := img.(*image.NRGBA64)
		for y := 0; y < dstH; y++ {
			for x := 0; x < dstW; x++ {
				got := n64.NRGBA64At(x, y)
				if absDiffU16(got.R, want.R) > 2 || absDiffU16(got.G, want.G) > 2 || absDiffU16(got.B, want.B) > 2 {
					t.Fatalf("at (%d,%d): got %v, want %v", x, y, got, want)
				}
				if absDiffU16(got.A, want.A) > 2 {
					t.Fatalf("at (%d,%d) A: got %d, want %d", x, y, got.A, want.A)
				}
			}
		}
	})

	t.Run("PalettedP8", func(t *testing.T) {
		pal := color.Palette{
			color.RGBA{R: 200, G: 100, B: 50, A: 255},
			color.RGBA{R: 0, G: 0, B: 255, A: 255}, // second entry, never used
		}
		src := image.NewPaletted(image.Rect(0, 0, srcW, srcH), pal)
		for y := 0; y < srcH; y++ {
			for x := 0; x < srcW; x++ {
				src.SetColorIndex(x, y, 0) // all index 0 = (200,100,50,255)
			}
		}
		img := mustShrink(t, src, dstW, dstH)
		nrgba := img.(*image.NRGBA)
		for y := 0; y < dstH; y++ {
			for x := 0; x < dstW; x++ {
				got := nrgba.NRGBAAt(x, y)
				if absDiffU8(got.R, 200) > 1 || absDiffU8(got.G, 100) > 1 || absDiffU8(got.B, 50) > 1 || got.A != 255 {
					t.Fatalf("at (%d,%d): got %v, want approx NRGBA{200,100,50,255}", x, y, got)
				}
			}
		}
	})

	t.Run("PalettedP8_TransparentEntry", func(t *testing.T) {
		// Palette with semi-transparent first entry; full fill with index 0.
		pal := color.Palette{
			color.NRGBA{R: 80, G: 160, B: 40, A: 128},
			color.RGBA{R: 255, G: 255, B: 255, A: 255},
		}
		src := image.NewPaletted(image.Rect(0, 0, srcW, srcH), pal)
		for y := 0; y < srcH; y++ {
			for x := 0; x < srcW; x++ {
				src.SetColorIndex(x, y, 0)
			}
		}
		img := mustShrink(t, src, dstW, dstH)
		nrgba := img.(*image.NRGBA)
		for y := 0; y < dstH; y++ {
			for x := 0; x < dstW; x++ {
				got := nrgba.NRGBAAt(x, y)
				if absDiffU8(got.R, 80) > 1 || absDiffU8(got.G, 160) > 1 ||
					absDiffU8(got.B, 40) > 1 || absDiffU8(got.A, 128) > 1 {
					t.Fatalf("at (%d,%d): got %v, want approx NRGBA{80,160,40,128}", x, y, got)
				}
			}
		}
	})
}

// TestShrinkOnlyHorizontal and TestShrinkOnlyVertical test partial scaling
// (one axis at a time).
func TestShrinkOnlyHorizontal(t *testing.T) {
	src := image.NewGray16(image.Rect(0, 0, 64, 64))
	const wantY = uint16(0xc000)
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			src.SetGray16(x, y, color.Gray16{Y: wantY})
		}
	}
	var buf bytes.Buffer
	require.NoError(t, png.Encode(&buf, src))

	// targetHeight=0 means no vertical shrink.
	img, err := Decode(bytes.NewReader(buf.Bytes()), 32, 0, MitchellNetravali)
	require.NoError(t, err)
	require.Equal(t, image.Rect(0, 0, 32, 64), img.Bounds())
}

func TestShrinkOnlyVertical(t *testing.T) {
	src := image.NewGray16(image.Rect(0, 0, 64, 64))
	const wantY = uint16(0xc000)
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			src.SetGray16(x, y, color.Gray16{Y: wantY})
		}
	}
	var buf bytes.Buffer
	require.NoError(t, png.Encode(&buf, src))

	// targetWidth=0 means no horizontal shrink.
	img, err := Decode(bytes.NewReader(buf.Bytes()), 0, 32, MitchellNetravali)
	require.NoError(t, err)
	require.Equal(t, image.Rect(0, 0, 64, 32), img.Bounds())
}

// mustShrink encodes src as PNG and decodes it through the scaled decoder.
func mustShrink(t *testing.T, src image.Image, dstW, dstH int) image.Image {
	t.Helper()
	var buf bytes.Buffer
	require.NoError(t, png.Encode(&buf, src))
	img, err := Decode(bytes.NewReader(buf.Bytes()), dstW, dstH, MitchellNetravali)
	require.NoError(t, err)
	require.Equal(t, image.Rect(0, 0, dstW, dstH), img.Bounds())
	return img
}

func absDiffU16(a, b uint16) int {
	d := int(a) - int(b)
	if d < 0 {
		return -d
	}
	return d
}

func absDiffU8(a, b uint8) int {
	d := int(a) - int(b)
	if d < 0 {
		return -d
	}
	return d
}
