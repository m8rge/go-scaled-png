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

// One flat table per (srcW, dstW, filter.Support). Kernel pointer equality is unreliable,
// so we key by support; if you change Kernel impl with same support, rebuild explicitly.
type coeffTable struct {
	srcW, dstW int
	scale      float64 // = float64(srcW)/float64(dstW)
	support    float64 // = filter.Support

	left []int32   // len == dstW, starting src index for each dst pixel
	off  []int32   // len == dstW, offset into w[] where weights start
	cnt  []uint16  // len == dstW, number of taps for this dst pixel
	w    []float32 // flat weights buffer (all pixels concatenated)
}

func buildCoeffTableFlat(srcW, dstW int, f Filter) *coeffTable {
	ct := &coeffTable{
		srcW: srcW, dstW: dstW,
		scale:   float64(srcW) / float64(dstW),
		support: f.Support,
		left:    make([]int32, dstW),
		off:     make([]int32, dstW),
		cnt:     make([]uint16, dstW),
	}

	// First pass: figure total number of weights to allocate.
	total := 0
	supp := f.Support * ct.scale
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
		total += taps
	}

	ct.w = make([]float32, total)

	// Second pass: fill metadata + weights (normalized per pixel).
	wpos := 0
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

		ct.left[x] = int32(left)
		ct.off[x] = int32(wpos)
		ct.cnt[x] = uint16(taps)

		var sum float64
		for i := 0; i < taps; i++ {
			si := left + i
			// normalized distance in source domain
			dx := (float64(si) + 0.5 - center) / ct.scale
			ww := f.Kernel(dx)
			ct.w[wpos+i] = float32(ww)
			sum += ww
		}
		if sum == 0 {
			// Fallback: nearest neighbor
			// collapse to one tap at nearest source pixel
			sx := int(center)
			if sx < 0 {
				sx = 0
			} else if sx >= srcW {
				sx = srcW - 1
			}
			ct.left[x] = int32(sx)
			ct.off[x] = int32(wpos)
			ct.cnt[x] = 1
			ct.w[wpos] = 1
			wpos += 1
			continue
		}
		inv := float32(1.0 / sum)
		for i := 0; i < taps; i++ {
			ct.w[wpos+i] *= inv
		}
		wpos += taps
	}
	return ct
}

var (
	coeffCacheMu sync.Mutex
	coeffCache   struct {
		srcW, dstW int
		support    float64
		ct         *coeffTable
	}
)

func getCoeffTable(srcW, dstW int, f Filter) *coeffTable {
	coeffCacheMu.Lock()
	c := coeffCache
	if c.ct != nil && c.srcW == srcW && c.dstW == dstW && c.support == f.Support {
		ct := c.ct
		coeffCacheMu.Unlock()
		return ct
	}
	coeffCacheMu.Unlock()

	ct := buildCoeffTableFlat(srcW, dstW, f)

	coeffCacheMu.Lock()
	coeffCache = struct {
		srcW, dstW int
		support    float64
		ct         *coeffTable
	}{srcW, dstW, f.Support, ct}
	coeffCacheMu.Unlock()
	return ct
}

// Gray -> Gray
func resampleGrayRowInto(dst []byte, dstW int, src []byte, srcW int, f Filter) {
	if dstW <= 0 || srcW <= 0 {
		return
	}
	ct := getCoeffTable(srcW, dstW, f)
	for x := 0; x < dstW; x++ {
		left := int(ct.left[x])
		off := int(ct.off[x])
		n := int(ct.cnt[x])
		ws := ct.w[off : off+n]

		var acc float32
		for i := 0; i < n; i++ {
			acc += float32(src[left+i]) * ws[i]
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
		left := int(ct.left[x])
		off := int(ct.off[x])
		n := int(ct.cnt[x])
		ws := ct.w[off : off+n]

		var r, g, b float32
		si := left * 3
		for i := 0; i < n; i++ {
			w := ws[i]
			r += float32(src[si+0]) * w
			g += float32(src[si+1]) * w
			b += float32(src[si+2]) * w
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

// Premultiplied RGBA -> RGBA (correct alpha handling)
func resampleRGBAPremulInto(dst []byte, dstW int, src []byte, srcW int, f Filter) {
	if dstW <= 0 || srcW <= 0 {
		return
	}
	ct := getCoeffTable(srcW, dstW, f)
	for x := 0; x < dstW; x++ {
		left := int(ct.left[x])
		off := int(ct.off[x])
		n := int(ct.cnt[x])
		ws := ct.w[off : off+n]

		var ar, ag, ab, aa float32
		si := left * 4
		for i := 0; i < n; i++ {
			w := ws[i]
			a := float32(src[si+3])
			aa += a * w
			aw := a * w
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
		left := int(ct.left[x])
		off := int(ct.off[x])
		n := int(ct.cnt[x])
		ws := ct.w[off : off+n]

		var v float32
		for i := 0; i < n; i++ {
			v += float32(src[left+i]) * ws[i]
		}
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		iv := byte(v + 0.5)
		i := x * 4
		dst[i+0] = iv
		dst[i+1] = iv
		dst[i+2] = iv
		dst[i+3] = alpha
	}
}
