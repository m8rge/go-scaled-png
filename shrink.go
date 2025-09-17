package pngscaled

import (
	"math"
	"sync"
)

// --- Optional horizontal shrinking API (minimal-invasive) --------------------
// TargetWidth enables optional horizontal shrinking during decode.
// If 0 or >= source width, shrinking is disabled (original behavior).
// MVP: only 8-bit decode paths (cbTC8 / cbTCA8 / cbG8) are supported here.
var TargetWidth int

// Filter is a generic resampling filter: identical in spirit to imaging.ResampleFilter.
type Filter struct {
	Support float64
	Kernel  func(x float64) float64
}

// TargetFilter, if non-nil, selects the resampling kernel to use for horizontal shrink.
// Example adapter (if you use github.com/disintegration/imaging or kovidgoyal/imaging):
//
//	png.UseImagingFilter(imaging.Lanczos)
var TargetFilter *Filter

// UseImagingFilter adapts an imaging.ResampleFilter (Support, Kernel func) to our Filter.
// Declare it here to avoid importing "imaging" and keep upstream diffs tiny.
// Call from your code like: png.UseImagingFilter(imaging.Lanczos)
func UseImagingFilter(support float64, kernel func(float64) float64) {
	TargetFilter = &Filter{Support: support, Kernel: kernel}
}

// --- Coeff table -------------------------------------------------------------

type coeffRow struct {
	left int
	w    []float32 // length == taps for this dst pixel (normalized)
}

type coeffTable struct {
	srcW, dstW int
	filter     Filter
	scale      float64
	supportPx  int
	rows       []coeffRow
}

// Build once per (srcW,dstW,filter). Anti-aliasing: taps grow with shrink factor.
func buildCoeffTable(srcW, dstW int, f Filter) *coeffTable {
	ct := &coeffTable{
		srcW: srcW, dstW: dstW, filter: f,
		scale: float64(srcW) / float64(dstW),
	}
	// Effective support in *source* pixels
	supp := f.Support * ct.scale
	// Max taps per pixel ~ floor(2*supp)+1
	maxTaps := int(math.Floor(2*supp)) + 1
	if maxTaps < 1 {
		maxTaps = 1
	}
	ct.supportPx = maxTaps

	ct.rows = make([]coeffRow, dstW)
	for x := 0; x < dstW; x++ {
		center := (float64(x) + 0.5) * ct.scale
		left := int(math.Ceil(center - supp))
		right := int(math.Floor(center + supp))
		if left < 0 {
			left = 0
		}
		if right >= srcW {
			right = srcW - 1
		}
		taps := right - left + 1
		if taps < 1 {
			taps = 1
		}

		// compute weights & normalize
		w := make([]float32, taps)
		var sum float64
		for i := 0; i < taps; i++ {
			// map tap index -> source pixel index
			si := left + i
			// normalized distance in source domain
			dx := (float64(si) + 0.5 - center) / ct.scale
			ww := f.Kernel(dx)
			sum += ww
			w[i] = float32(ww)
		}
		if sum == 0 {
			// fallback to NN
			for i := range w {
				w[i] = 0
			}
			sx := int(center)
			if sx < 0 {
				sx = 0
			} else if sx >= srcW {
				sx = srcW - 1
			}
			left = sx
			w = w[:1]
			w[0] = 1
		} else {
			inv := float32(1.0 / sum)
			for i := range w {
				w[i] *= inv
			}
		}
		ct.rows[x] = coeffRow{left: left, w: w}
	}
	return ct
}

// Small LRU-ish cache so we don't rebuild per row (optional, but handy).
var (
	coeffCacheMu sync.Mutex
	coeffCache   = struct {
		srcW, dstW int
		support    float64
		ptr        *coeffTable
	}{}
)

func getCoeffTable(srcW, dstW int, f Filter) *coeffTable {
	coeffCacheMu.Lock()
	defer coeffCacheMu.Unlock()

	c := coeffCache
	if c.ptr != nil && c.srcW == srcW && c.dstW == dstW && c.support == f.Support {
		ct := c.ptr
		return ct
	}

	ct := buildCoeffTable(srcW, dstW, f)

	coeffCache = struct {
		srcW, dstW int
		support    float64
		ptr        *coeffTable
	}{srcW, dstW, f.Support, ct}
	return ct
}

// Gray -> Gray
func resampleGrayRowInto(dst []byte, dstW int, src []byte, srcW int, f Filter) {
	if dstW <= 0 || srcW <= 0 {
		return
	}
	ct := getCoeffTable(srcW, dstW, f)
	for x := 0; x < dstW; x++ {
		cr := ct.rows[x]
		left := cr.left
		w := cr.w
		var acc float32
		for i := 0; i < len(w); i++ {
			acc += float32(src[left+i]) * w[i]
		}
		if acc < 0 {
			acc = 0
		} else if acc > 255 {
			acc = 255
		}
		dst[x] = byte(acc + 0.5)
	}
}

// RGB -> RGBA (alpha const)
func resampleRGBtoRGBAInto(dst []byte, dstW int, src []byte, srcW int, f Filter, alpha byte) {
	if dstW <= 0 || srcW <= 0 {
		return
	}
	ct := getCoeffTable(srcW, dstW, f)
	for x := 0; x < dstW; x++ {
		cr := ct.rows[x]
		left := cr.left
		w := cr.w
		var r, g, b float32
		si := left * 3
		for i := 0; i < len(w); i++ {
			ww := w[i]
			r += float32(src[si+0]) * ww
			g += float32(src[si+1]) * ww
			b += float32(src[si+2]) * ww
			si += 3
		}
		i := x * 4
		if r < 0 {
			r = 0
		} else if r > 255 {
			r = 255
		}
		if g < 0 {
			g = 0
		} else if g > 255 {
			g = 255
		}
		if b < 0 {
			b = 0
		} else if b > 255 {
			b = 255
		}
		dst[i+0] = byte(r + 0.5)
		dst[i+1] = byte(g + 0.5)
		dst[i+2] = byte(b + 0.5)
		dst[i+3] = alpha
	}
}

// resampleRGBAStraightInto: RGBA resampling WITHOUT premultiplying.
// Uses the cached coeff table; no allocations in the hot path.
// Expected: can produce halos on semi-transparent edges vs the premul version.
func resampleRGBAStraightInto(dst []byte, dstW int, src []byte, srcW int, f Filter) {
	if dstW <= 0 || srcW <= 0 {
		return
	}
	const chans = 4
	ct := getCoeffTable(srcW, dstW, f)
	for x := 0; x < dstW; x++ {
		cr := ct.rows[x]
		left := cr.left
		w := cr.w

		var r, g, b, a float32
		si := left * chans
		for i := 0; i < len(w); i++ {
			ww := w[i]
			r += float32(src[si+0]) * ww
			g += float32(src[si+1]) * ww
			b += float32(src[si+2]) * ww
			a += float32(src[si+3]) * ww
			si += chans
		}
		if r < 0 {
			r = 0
		} else if r > 255 {
			r = 255
		}
		if g < 0 {
			g = 0
		} else if g > 255 {
			g = 255
		}
		if b < 0 {
			b = 0
		} else if b > 255 {
			b = 255
		}
		if a < 0 {
			a = 0
		} else if a > 255 {
			a = 255
		}
		di := x * chans
		dst[di+0] = byte(r + 0.5)
		dst[di+1] = byte(g + 0.5)
		dst[di+2] = byte(b + 0.5)
		dst[di+3] = byte(a + 0.5)
	}
}

// Premultiplied RGBA -> RGBA (correct alpha handling)
func resampleRGBAPremulInto(dst []byte, dstW int, src []byte, srcW int, f Filter) {
	if dstW <= 0 || srcW <= 0 {
		return
	}
	ct := getCoeffTable(srcW, dstW, f)
	for x := 0; x < dstW; x++ {
		cr := ct.rows[x]
		left := cr.left
		w := cr.w
		var ar, ag, ab, aa float32
		si := left * 4
		for i := 0; i < len(w); i++ {
			ww := w[i]
			a := float32(src[si+3])
			aa += a * ww
			aw := a * ww
			ar += float32(src[si+0]) * aw
			ag += float32(src[si+1]) * aw
			ab += float32(src[si+2]) * aw
			si += 4
		}
		var R, G, B float32
		if aa > 0 {
			invA := 1.0 / aa
			R, G, B = ar*invA, ag*invA, ab*invA
		}
		if R < 0 {
			R = 0
		} else if R > 255 {
			R = 255
		}
		if G < 0 {
			G = 0
		} else if G > 255 {
			G = 255
		}
		if B < 0 {
			B = 0
		} else if B > 255 {
			B = 255
		}
		if aa < 0 {
			aa = 0
		} else if aa > 255 {
			aa = 255
		}
		i := x * 4
		dst[i+0] = byte(R + 0.5)
		dst[i+1] = byte(G + 0.5)
		dst[i+2] = byte(B + 0.5)
		dst[i+3] = byte(aa + 0.5)
	}
}

// Gray (1 chan) -> RGBA (4 chan) with constant alpha.
// dst must have capacity >= dstW*4. src is len == srcW.
// Uses the prebuilt coeff table; no per-pixel allocs.
func resampleGrayToRGBAInto(dst []byte, dstW int, src []byte, srcW int, f Filter, alpha byte) {
	if dstW <= 0 || srcW <= 0 {
		return
	}
	ct := getCoeffTable(srcW, dstW, f)
	for x := 0; x < dstW; x++ {
		cr := ct.rows[x]
		left := cr.left
		w := cr.w

		var v float32
		for i := 0; i < len(w); i++ {
			v += float32(src[left+i]) * w[i]
		}
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		iv := byte(v + 0.5)
		di := x * 4
		dst[di+0] = iv // R
		dst[di+1] = iv // G
		dst[di+2] = iv // B
		dst[di+3] = alpha
	}
}
