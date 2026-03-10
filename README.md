# go-scaled-png

A PNG decoder that scales images down **while decoding**, avoiding the
memory cost of allocating a full-size intermediate image.

## Installation

```bash
go get github.com/m8rge/go-scaled-png
```

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
import pngscaled "github.com/m8rge/go-scaled-png"

img, err := pngscaled.Decode(r, targetWidth, targetHeight, pngscaled.MitchellNetravali)
```

- `r` — any `io.Reader` over PNG data.
- `targetWidth`, `targetHeight` — desired output dimensions. Pass `0` to
  skip resizing on that axis (the original dimension is preserved).
- Filter — one of the supported kernels (see below).

## Filters

`MitchellNetravali` is the recommended default; `CatmullRom` and `Lanczos`
are sharper; `Linear` and `Box` are faster; and `Gaussian`, `BSpline`,
`Hermite`, `Bartlett`, `Hann`, `Hamming`, `Blackman`, `Welch`, and `Cosine`
are available for specialized needs.

## Limitations

- **Upscaling is not supported.** If `targetWidth` or `targetHeight` is
  larger than the source dimension on that axis, the source size is used
  instead.
- **Interlaced PNG images are not resized.** The full image is decoded and
  returned at its original size.

## Benchmarks

Benchmarks for a `3456x2234` source image (16" MBP screen resolution) are in
`BenchmarkResizeLargePNGScaled` and `BenchmarkResizeLargeStdlibImaging`:

```bash
go test -run '^$' -bench 'BenchmarkResizeLarge(PNGScaled|StdlibImaging)' -benchmem -count=5 ./...
```

These cover `3456x2234 -> 1920x1080` (Full HD) and `3456x2234 -> 32x32`.

Reference run (`darwin/arm64`, Apple M2, averages over `-count=5`):

| Benchmark | CPU (ns/op) | Memory (B/op) | Allocs (allocs/op) |
|---|---:|---:|---:|
| `BenchmarkResizeLargePNGScaled/3456x2234_to_1920x1080` | `74423030` | `17274031` | `35.8` |
| `BenchmarkResizeLargePNGScaled/3456x2234_to_32x32` | `55202217` | `354517` | `16.0` |

For baseline comparison (`png.Decode` + `imaging.Resize` on the same data):

| Benchmark | CPU (ns/op) | Memory (B/op) | Allocs (allocs/op) |
|---|---:|---:|---:|
| `BenchmarkResizeLargeStdlibImaging/3456x2234_to_1920x1080` | `58284645` | `57329325` | `63.0` |
| `BenchmarkResizeLargeStdlibImaging/3456x2234_to_32x32` | `45089980` | `31831211` | `63.0` |

The sample visual inspection tool is documented in
`cmd/shrink-samples/README.md`.

## Testing

```
go test ./...
go test -run TestShrinkGolden -update   # regenerate golden reference files
```

Tests use the [public-domain PNG test suite][pngsuite] files in
`testdata/pngsuite/`, covering all PNG color types, bit depths, filter
types, and tRNS variants.

[pngsuite]: http://www.schaik.com/pngsuite/
