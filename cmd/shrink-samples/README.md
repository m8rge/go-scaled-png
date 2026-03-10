# shrink-samples

Utility for visual inspection of resized PNG outputs.

## Usage

Run from repository root:

```bash
go run ./cmd/shrink-samples [-out dir] [-w N] [-h N] [files...]
```

If no files are provided, the tool processes all `testdata/pngsuite/*.png`
images and writes outputs to `out/`.

## Flags

- `-out` output directory (default `out`)
- `-w` target width in pixels (default `16`)
- `-h` target height in pixels (default `16`)

## Example

```bash
go run ./cmd/shrink-samples -out /tmp/pngscaled -w 1920 -h 1080 testdata/benchRGB.png
```
