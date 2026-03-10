# go-scaled-png

A PNG decoder that scales images down **while decoding**, avoiding the
memory cost of allocating a full-size intermediate image.

## How it works

The standard approach — `png.Decode` then `imaging.Resize` — allocates the
full-resolution image first, then resamples it. For large PNGs this means
holding two images in memory simultaneously.

This package integrates the horizontal resample pass directly into the PNG
row-decode loop. Each source row is resampled horizontally as it comes out of
the DEFLATE stream. The vertical resample pass runs in-place after all rows
are decoded, using the pre-allocated destination buffer.

Peak memory usage is proportional to the **output** image size, not the
input size.

## API

```go
import pngscaled "go-scaled-png"

img, err := pngscaled.Decode(r, targetWidth, targetHeight, pngscaled.MitchellNetravali)
```

- `r` — any `io.Reader` over PNG data.
- `targetWidth`, `targetHeight` — desired output dimensions. Pass `0` to
  skip resizing on that axis (the original dimension is preserved).
- Filter — one of the filters listed below.

The returned `image.Image` has the concrete type that matches the PNG color
type:

| PNG color type | Returned type |
|---|---|
| Gray 1/2/4/8-bit, no tRNS | `*image.Gray` |
| Gray 16-bit, no tRNS | `*image.Gray16` |
| Gray-Alpha 8-bit | `*image.NRGBA` |
| Gray-Alpha 16-bit | `*image.NRGBA64` |
| Gray + tRNS (any depth) | `*image.NRGBA` / `*image.NRGBA64` |
| Truecolor 8-bit, no tRNS | `*image.RGBA` |
| Truecolor 16-bit, no tRNS | `*image.RGBA64` |
| Truecolor 8-bit + tRNS | `*image.NRGBA` |
| Truecolor 16-bit + tRNS | `*image.NRGBA64` |
| Truecolor-Alpha 8-bit | `*image.NRGBA` |
| Truecolor-Alpha 16-bit | `*image.NRGBA64` |
| Paletted (any depth) | `*image.NRGBA` (shrink path) / `*image.Paletted` (no resize) |
| Interlaced | original type, not resized |

## Filters

| Filter | Notes |
|---|---|
| `MitchellNetravali` | Good default: smooth, minimal ringing |
| `CatmullRom` | Sharper than Mitchell, similar to Lanczos |
| `Lanczos` | Highest quality for photographic images |
| `Linear` | Fast bilinear |
| `Box` | Simple averaging, fastest |
| `Gaussian`, `BSpline`, `Hermite`, `Bartlett`, `Hann`, `Hamming`, `Blackman`, `Welch`, `Cosine` | Specialist use |

## Transparency (tRNS)

PNG encodes transparency either as an alpha channel (color types GA, RGBA)
or as a single transparent color value (tRNS chunk). Both cases are handled
with premultiplied-alpha resampling to avoid dark halos at transparent
boundaries. The output image stores straight (non-premultiplied) alpha.

## Limitations

- **Upscaling is not supported.** If `targetWidth` or `targetHeight` is
  larger than the source dimension on that axis, the source size is used
  instead.
- **Interlaced PNG images are not resized.** The full image is decoded and
  returned at its original size.

## Visual inspection tool

```
go run ./cmd/shrink-samples [-out dir] [-w N] [-h N] [files...]
```

Decodes and scales PNG files, writing results to the output directory.
Defaults to all `testdata/pngsuite/*.png` files at 16×16.

## Testing

```
go test ./...
go test -run TestShrinkGolden -update   # regenerate golden reference files
```

Tests use the [public-domain PNG test suite][pngsuite] files in
`testdata/pngsuite/`, covering all PNG color types, bit depths, filter
types, and tRNS variants.

[pngsuite]: http://www.schaik.com/pngsuite/
