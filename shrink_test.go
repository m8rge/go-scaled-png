package pngscaled

import (
	"image/png"
	"os"
	"runtime"
	"testing"

	"github.com/kovidgoyal/imaging"
	"github.com/stretchr/testify/require"
)

func TestShrink(t *testing.T) {
	TargetWidth = 400
	UseImagingFilter(imaging.Lanczos.Support, imaging.Lanczos.Kernel)

	names := []string{"1.png", "2.png", "3.png"}
	outNames := []string{"1-out.png", "2-out.png", "3-out.png"}

	for i, name := range names {
		data, err := os.Open(name)
		require.NoError(t, err)
		image, err := Decode(data)
		require.NoError(t, err)

		file, err := os.Create(outNames[i])
		require.NoError(t, err)
		err = png.Encode(file, image)
		require.NoError(t, err)
	}
}

func BenchmarkShrink(b *testing.B) {
	runtime.GOMAXPROCS(1)
	TargetWidth = 400
	UseImagingFilter(imaging.Lanczos.Support, imaging.Lanczos.Kernel)

	names := []string{"1.png"}
	files := make([]*os.File, len(names))
	for i, name := range names {
		data, err := os.Open(name)
		require.NoError(b, err)
		files[i] = data
	}

	b.ReportAllocs()
	for b.Loop() {
		for _, file := range files {
			file.Seek(0, 0)

			_, err := Decode(file)
			require.NoError(b, err)
		}
	}
}

func BenchmarkShrinkInitial(b *testing.B) {
	runtime.GOMAXPROCS(1)
	TargetWidth = 0

	names := []string{"1.png"}
	files := make([]*os.File, len(names))
	for i, name := range names {
		data, err := os.Open(name)
		require.NoError(b, err)
		files[i] = data
	}

	b.ReportAllocs()
	for b.Loop() {
		for _, file := range files {
			file.Seek(0, 0)

			image, err := png.Decode(file)
			require.NoError(b, err)
			_ = imaging.Resize(image, 400, image.Bounds().Dy(), imaging.Lanczos)
		}
	}
}
