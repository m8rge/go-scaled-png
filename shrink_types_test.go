package pngscaled

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

var update = flag.Bool("update", false, "regenerate golden files in testdata/golden/")

// TestShrinkPNGSuiteTypes verifies that shrinking works for all PNG test suite
// files without panics, and returns the correct Go image type and bounds.
// Files are from the public-domain PNG test suite shipped in testdata/pngsuite/.
func TestShrinkPNGSuiteTypes(t *testing.T) {
	dstRect := image.Rect(0, 0, 16, 16) // expected bounds for all resizable files
	cases := []struct {
		file      string
		wantType  string      // expected fmt.Sprintf("%T", img)
		wantBounds image.Rectangle // zero value → use dstRect
	}{
		// sub-byte gray, no tRNS
		{"basn0g01", "*image.Gray", image.Rectangle{}},  // cbG1, 32×32 → Gray
		{"basn0g02", "*image.Gray", image.Rectangle{}},  // cbG2, 32×32 → Gray
		{"basn0g04", "*image.Gray", image.Rectangle{}},  // cbG4, 32×32 → Gray
		// sub-byte gray with non-standard widths (29/30/31 pixels)
		{"basn0g01-30", "*image.Gray", image.Rectangle{}}, // cbG1, 30×30 → Gray
		{"basn0g02-29", "*image.Gray", image.Rectangle{}}, // cbG2, 29×29 → Gray
		{"basn0g04-31", "*image.Gray", image.Rectangle{}}, // cbG4, 31×31 → Gray
		// sub-byte gray + tRNS
		{"ftbbn0g01", "*image.NRGBA", image.Rectangle{}}, // cbG1 + tRNS → NRGBA
		{"ftbbn0g02", "*image.NRGBA", image.Rectangle{}}, // cbG2 + tRNS → NRGBA
		{"ftbbn0g04", "*image.NRGBA", image.Rectangle{}}, // cbG4 + tRNS → NRGBA
		// 8-bit types
		{"basn0g08", "*image.Gray", image.Rectangle{}},   // cbG8, no tRNS → Gray
		{"basn4a08", "*image.NRGBA", image.Rectangle{}},  // cbGA8          → NRGBA
		{"basn2c08", "*image.RGBA", image.Rectangle{}},   // cbTC8, no tRNS → RGBA
		{"ftbrn2c08", "*image.NRGBA", image.Rectangle{}}, // cbTC8 + tRNS   → NRGBA
		{"basn6a08", "*image.NRGBA", image.Rectangle{}},  // cbTCA8         → NRGBA
		// PNG filter type tests (same color types, different filter encoding)
		{"ftp0n0g08", "*image.Gray", image.Rectangle{}},  // cbG8, filter0  → Gray
		{"ftp0n2c08", "*image.RGBA", image.Rectangle{}},  // cbTC8, filter0 → RGBA
		{"ftp0n3p08", "*image.NRGBA", image.Rectangle{}}, // cbP8, filter0  → NRGBA
		{"ftp1n3p08", "*image.NRGBA", image.Rectangle{}}, // cbP8, filter1  → NRGBA
		// 16-bit types
		{"basn0g16", "*image.Gray16", image.Rectangle{}},   // cbG16, no tRNS   → Gray16
		{"ftbwn0g16", "*image.NRGBA64", image.Rectangle{}}, // cbG16 + tRNS     → NRGBA64
		{"basn4a16", "*image.NRGBA64", image.Rectangle{}},  // cbGA16           → NRGBA64
		{"basn2c16", "*image.RGBA64", image.Rectangle{}},   // cbTC16, no tRNS  → RGBA64
		{"ftbbn2c16", "*image.NRGBA64", image.Rectangle{}}, // cbTC16+tRNS (black) → NRGBA64
		{"ftbgn2c16", "*image.NRGBA64", image.Rectangle{}}, // cbTC16+tRNS (green) → NRGBA64
		{"basn6a16", "*image.NRGBA64", image.Rectangle{}},  // cbTCA16          → NRGBA64
		// paletted types
		{"basn3p01", "*image.NRGBA", image.Rectangle{}},      // cbP1        → NRGBA
		{"basn3p02", "*image.NRGBA", image.Rectangle{}},      // cbP2        → NRGBA
		{"basn3p04", "*image.NRGBA", image.Rectangle{}},      // cbP4        → NRGBA
		{"basn3p08", "*image.NRGBA", image.Rectangle{}},      // cbP8        → NRGBA
		{"basn3p08-trns", "*image.NRGBA", image.Rectangle{}}, // cbP8 + tRNS → NRGBA
		{"ftbbn3p08", "*image.NRGBA", image.Rectangle{}},     // cbP8 + tRNS (black bKGD) → NRGBA
		{"ftbgn3p08", "*image.NRGBA", image.Rectangle{}},     // cbP8 + tRNS (green bKGD) → NRGBA
		{"ftbwn3p08", "*image.NRGBA", image.Rectangle{}},     // cbP8 + tRNS (white bKGD) → NRGBA
		{"ftbyn3p08", "*image.NRGBA", image.Rectangle{}},     // cbP8 + tRNS (yellow bKGD) → NRGBA
		// interlaced paletted — resize is skipped, returned at original size
		{"basn3p04-31i", "*image.Paletted", image.Rect(0, 0, 31, 31)}, // cbP4, interlaced → Paletted 31×31
	}
	for _, tc := range cases {
		t.Run(tc.file, func(t *testing.T) {
			f, err := os.Open("testdata/pngsuite/" + tc.file + ".png")
			require.NoError(t, err)
			defer f.Close()

			img, err := Decode(f, 16, 16, MitchellNetravali)
			require.NoError(t, err)

			wantBounds := tc.wantBounds
			if wantBounds == (image.Rectangle{}) {
				wantBounds = dstRect
			}
			require.Equal(t, wantBounds, img.Bounds(),
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

	t.Run("Gray8", func(t *testing.T) {
		const wantY = uint8(0x7e)
		src := image.NewGray(image.Rect(0, 0, srcW, srcH))
		for y := 0; y < srcH; y++ {
			for x := 0; x < srcW; x++ {
				src.SetGray(x, y, color.Gray{Y: wantY})
			}
		}
		img := mustShrink(t, src, dstW, dstH)
		gray := img.(*image.Gray)
		for y := 0; y < dstH; y++ {
			for x := 0; x < dstW; x++ {
				got := gray.GrayAt(x, y).Y
				if absDiffU8(got, wantY) > 1 {
					t.Fatalf("at (%d,%d): got %d, want %d", x, y, got, wantY)
				}
			}
		}
	})

	t.Run("NRGBA_GA8_semitransparent", func(t *testing.T) {
		// image.NewNRGBA with A < 0xff encodes as cbGA8.
		want := color.NRGBA{R: 120, G: 120, B: 120, A: 180}
		src := image.NewNRGBA(image.Rect(0, 0, srcW, srcH))
		for y := 0; y < srcH; y++ {
			for x := 0; x < srcW; x++ {
				src.SetNRGBA(x, y, want)
			}
		}
		img := mustShrink(t, src, dstW, dstH)
		nrgba := img.(*image.NRGBA)
		for y := 0; y < dstH; y++ {
			for x := 0; x < dstW; x++ {
				got := nrgba.NRGBAAt(x, y)
				if absDiffU8(got.R, want.R) > 1 || absDiffU8(got.G, want.G) > 1 ||
					absDiffU8(got.B, want.B) > 1 || absDiffU8(got.A, want.A) > 1 {
					t.Fatalf("at (%d,%d): got %v, want approx %v", x, y, got, want)
				}
			}
		}
	})

	t.Run("RGBA_TC8_fullalpha", func(t *testing.T) {
		// image.NewNRGBA with all A=0xff encodes as cbTC8 → returns *image.RGBA.
		want := color.NRGBA{R: 200, G: 80, B: 40, A: 0xff}
		src := image.NewNRGBA(image.Rect(0, 0, srcW, srcH))
		for y := 0; y < srcH; y++ {
			for x := 0; x < srcW; x++ {
				src.SetNRGBA(x, y, want)
			}
		}
		img := mustShrink(t, src, dstW, dstH)
		rgba := img.(*image.RGBA)
		for y := 0; y < dstH; y++ {
			for x := 0; x < dstW; x++ {
				got := rgba.RGBAAt(x, y)
				if absDiffU8(got.R, want.R) > 1 || absDiffU8(got.G, want.G) > 1 ||
					absDiffU8(got.B, want.B) > 1 || got.A != 0xff {
					t.Fatalf("at (%d,%d): got %v, want approx %v", x, y, got, want)
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

// TestShrinkGray8Transparent verifies the cbG8+tRNS shrink path (resampleGrayToRGBAIntoQ15).
// The binary is the same 15×11 gray+tRNS PNG used in TestGray8Transparent.
func TestShrinkGray8Transparent(t *testing.T) {
	data := []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
		0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x0b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x85, 0x2c, 0x88,
		0x80, 0x00, 0x00, 0x00, 0x02, 0x74, 0x52, 0x4e, 0x53, 0x00, 0xff, 0x5b, 0x91, 0x22, 0xb5, 0x00,
		0x00, 0x00, 0x02, 0x62, 0x4b, 0x47, 0x44, 0x00, 0xff, 0x87, 0x8f, 0xcc, 0xbf, 0x00, 0x00, 0x00,
		0x09, 0x70, 0x48, 0x59, 0x73, 0x00, 0x00, 0x0a, 0xf0, 0x00, 0x00, 0x0a, 0xf0, 0x01, 0x42, 0xac,
		0x34, 0x98, 0x00, 0x00, 0x00, 0x07, 0x74, 0x49, 0x4d, 0x45, 0x07, 0xd5, 0x04, 0x02, 0x12, 0x11,
		0x11, 0xf7, 0x65, 0x3d, 0x8b, 0x00, 0x00, 0x00, 0x4f, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63,
		0xf8, 0xff, 0xff, 0xff, 0xb9, 0xbd, 0x70, 0xf0, 0x8c, 0x01, 0xc8, 0xaf, 0x6e, 0x99, 0x02, 0x05,
		0xd9, 0x7b, 0xc1, 0xfc, 0x6b, 0xff, 0xa1, 0xa0, 0x87, 0x30, 0xff, 0xd9, 0xde, 0xbd, 0xd5, 0x4b,
		0xf7, 0xee, 0xfd, 0x0e, 0xe3, 0xef, 0xcd, 0x06, 0x19, 0x14, 0xf5, 0x1e, 0xce, 0xef, 0x01, 0x31,
		0x92, 0xd7, 0x82, 0x41, 0x31, 0x9c, 0x3f, 0x07, 0x02, 0xee, 0xa1, 0xaa, 0xff, 0xff, 0x9f, 0xe1,
		0xd9, 0x56, 0x30, 0xf8, 0x0e, 0xe5, 0x03, 0x00, 0xa9, 0x42, 0x84, 0x3d, 0xdf, 0x8f, 0xa6, 0x8f,
		0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
	}
	// Source is 15×11; shrink to 8×6.
	img, err := Decode(bytes.NewReader(data), 8, 6, MitchellNetravali)
	require.NoError(t, err)
	require.Equal(t, image.Rect(0, 0, 8, 6), img.Bounds())
	require.Equal(t, "*image.NRGBA", fmt.Sprintf("%T", img), "cbG8+tRNS must return NRGBA")
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

// TestShrinkGolden does a pixel-exact comparison against golden PNG files stored
// in testdata/golden/. Run with -update to regenerate the golden files.
//
//	go test -run TestShrinkGolden -update
func TestShrinkGolden(t *testing.T) {
	cases := []string{
		// sub-byte gray, no tRNS
		"basn0g01", "basn0g02", "basn0g04",
		// sub-byte gray with non-standard widths
		"basn0g01-30", "basn0g02-29", "basn0g04-31",
		// sub-byte gray + tRNS
		"ftbbn0g01", "ftbbn0g02", "ftbbn0g04",
		// 8-bit
		"basn0g08", "basn2c08", "basn4a08", "basn6a08", "ftbrn2c08",
		// PNG filter type tests
		"ftp0n0g08", "ftp0n2c08", "ftp0n3p08", "ftp1n3p08",
		// 16-bit
		"basn0g16", "ftbwn0g16", "basn2c16", "ftbbn2c16", "ftbgn2c16", "basn4a16", "basn6a16",
		// paletted
		"basn3p01", "basn3p02", "basn3p04", "basn3p08", "basn3p08-trns",
		"ftbbn3p08", "ftbgn3p08", "ftbwn3p08", "ftbyn3p08",
	}

	if *update {
		require.NoError(t, os.MkdirAll("testdata/golden", 0o755))
	}

	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			f, err := os.Open("testdata/pngsuite/" + name + ".png")
			require.NoError(t, err)
			defer f.Close()

			got, err := Decode(f, 16, 16, MitchellNetravali)
			require.NoError(t, err)

			goldenPath := "testdata/golden/" + name + ".png"

			if *update {
				out, err := os.Create(goldenPath)
				require.NoError(t, err)
				require.NoError(t, png.Encode(out, got))
				require.NoError(t, out.Close())
				return
			}

			gf, err := os.Open(goldenPath)
			require.NoError(t, err, "golden file missing – run: go test -run TestShrinkGolden -update")
			defer gf.Close()

			want, err := png.Decode(gf)
			require.NoError(t, err)

			require.Equal(t, want.Bounds(), got.Bounds(), "bounds mismatch for %s", name)
			assertImagesEqual(t, name, want, got)
		})
	}
}

// assertImagesEqual compares two images pixel-by-pixel via .RGBA() (premultiplied 16-bit).
func assertImagesEqual(t *testing.T, name string, want, got image.Image) {
	t.Helper()
	b := want.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			wr, wg, wb, wa := want.At(x, y).RGBA()
			gr, gg, gb, ga := got.At(x, y).RGBA()
			if wr != gr || wg != gg || wb != gb || wa != ga {
				t.Fatalf("%s: pixel mismatch at (%d,%d): want RGBA(%d,%d,%d,%d) got RGBA(%d,%d,%d,%d)",
					name, x, y,
					wr>>8, wg>>8, wb>>8, wa>>8,
					gr>>8, gg>>8, gb>>8, ga>>8)
			}
		}
	}
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
