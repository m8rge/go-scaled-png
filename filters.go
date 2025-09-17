// Copyright (c) 2012 Grigory Dryapak

package pngscaled

import (
	"math"
)

// ResampleFilter specifies a resampling filter to be used for image resizing.
//
//	General filter recommendations:
//
//	- Lanczos
//		A high-quality resampling filter for photographic images yielding sharp results.
//
//	- CatmullRom
//		A sharp cubic filter that is faster than Lanczos filter while providing similar results.
//
//	- MitchellNetravali
//		A cubic filter that produces smoother results with less ringing artifacts than CatmullRom.
//
//	- Linear
//		Bilinear resampling filter, produces a smooth output. Faster than cubic filters.
//
//	- Box
//		Simple and fast averaging filter appropriate for downscaling.
//		When upscaling it's similar to NearestNeighbor.
//
//	- NearestNeighbor
//		Fastest resampling filter, no antialiasing.
type ResampleFilter struct {
	Support float64
	Kernel  func(float64) float64
}

// NearestNeighbor is a nearest-neighbor filter (no anti-aliasing).
var NearestNeighbor ResampleFilter

// Box filter (averaging pixels).
var Box ResampleFilter

// Linear filter.
var Linear ResampleFilter

// Hermite cubic spline filter (BC-spline; B=0; C=0).
var Hermite ResampleFilter

// MitchellNetravali is Mitchell-Netravali cubic filter (BC-spline; B=1/3; C=1/3).
var MitchellNetravali ResampleFilter

// CatmullRom is a Catmull-Rom - sharp cubic filter (BC-spline; B=0; C=0.5).
var CatmullRom ResampleFilter

// BSpline is a smooth cubic filter (BC-spline; B=1; C=0).
var BSpline ResampleFilter

// Gaussian is a Gaussian blurring filter.
var Gaussian ResampleFilter

// Bartlett is a Bartlett-windowed sinc filter (3 lobes).
var Bartlett ResampleFilter

// Lanczos filter (3 lobes).
var Lanczos ResampleFilter

// Hann is a Hann-windowed sinc filter (3 lobes).
var Hann ResampleFilter

// Hamming is a Hamming-windowed sinc filter (3 lobes).
var Hamming ResampleFilter

// Blackman is a Blackman-windowed sinc filter (3 lobes).
var Blackman ResampleFilter

// Welch is a Welch-windowed sinc filter (parabolic window, 3 lobes).
var Welch ResampleFilter

// Cosine is a Cosine-windowed sinc filter (3 lobes).
var Cosine ResampleFilter

func bcspline(x, b, c float64) float64 {
	var y float64
	x = math.Abs(x)
	if x < 1.0 {
		y = ((12-9*b-6*c)*x*x*x + (-18+12*b+6*c)*x*x + (6 - 2*b)) / 6
	} else if x < 2.0 {
		y = ((-b-6*c)*x*x*x + (6*b+30*c)*x*x + (-12*b-48*c)*x + (8*b + 24*c)) / 6
	}
	return y
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	return math.Sin(math.Pi*x) / (math.Pi * x)
}

func init() {
	NearestNeighbor = ResampleFilter{
		Support: 0.0, // special case - not applying the filter
	}

	Box = ResampleFilter{
		Support: 0.5,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x <= 0.5 {
				return 1.0
			}
			return 0
		},
	}

	Linear = ResampleFilter{
		Support: 1.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 1.0 {
				return 1.0 - x
			}
			return 0
		},
	}

	Hermite = ResampleFilter{
		Support: 1.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 1.0 {
				return bcspline(x, 0.0, 0.0)
			}
			return 0
		},
	}

	MitchellNetravali = ResampleFilter{
		Support: 2.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 2.0 {
				return bcspline(x, 1.0/3.0, 1.0/3.0)
			}
			return 0
		},
	}

	CatmullRom = ResampleFilter{
		Support: 2.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 2.0 {
				return bcspline(x, 0.0, 0.5)
			}
			return 0
		},
	}

	BSpline = ResampleFilter{
		Support: 2.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 2.0 {
				return bcspline(x, 1.0, 0.0)
			}
			return 0
		},
	}

	Gaussian = ResampleFilter{
		Support: 2.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 2.0 {
				return math.Exp(-2 * x * x)
			}
			return 0
		},
	}

	Bartlett = ResampleFilter{
		Support: 3.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 3.0 {
				return sinc(x) * (3.0 - x) / 3.0
			}
			return 0
		},
	}

	Lanczos = ResampleFilter{
		Support: 3.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 3.0 {
				return sinc(x) * sinc(x/3.0)
			}
			return 0
		},
	}

	Hann = ResampleFilter{
		Support: 3.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 3.0 {
				return sinc(x) * (0.5 + 0.5*math.Cos(math.Pi*x/3.0))
			}
			return 0
		},
	}

	Hamming = ResampleFilter{
		Support: 3.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 3.0 {
				return sinc(x) * (0.54 + 0.46*math.Cos(math.Pi*x/3.0))
			}
			return 0
		},
	}

	Blackman = ResampleFilter{
		Support: 3.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 3.0 {
				return sinc(x) * (0.42 - 0.5*math.Cos(math.Pi*x/3.0+math.Pi) + 0.08*math.Cos(2.0*math.Pi*x/3.0))
			}
			return 0
		},
	}

	Welch = ResampleFilter{
		Support: 3.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 3.0 {
				return sinc(x) * (1.0 - (x * x / 9.0))
			}
			return 0
		},
	}

	Cosine = ResampleFilter{
		Support: 3.0,
		Kernel: func(x float64) float64 {
			x = math.Abs(x)
			if x < 3.0 {
				return sinc(x) * math.Cos((math.Pi/2.0)*(x/3.0))
			}
			return 0
		},
	}
}
