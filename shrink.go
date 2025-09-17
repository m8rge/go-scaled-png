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

// Q15 fixed-point weights table: one build per (srcW, dstW, filter.Support).
// We store weights in int16 where 1.0 == 32768 (1<<15). Per-pixel weights sum to 32768.
type coeffTableQ15 struct {
	srcW, dstW int
	scale      float64
	support    float64
	left       []int32  // len == dstW
	off        []int32  // len == dstW, offset into wQ15
	cnt        []uint16 // len == dstW, number of taps
	wQ15       []int16  // flat weights, all pixels concatenated
}

func buildCoeffTableFlatQ15(srcW, dstW int, f Filter) *coeffTableQ15 {
	ct := &coeffTableQ15{
		srcW: srcW, dstW: dstW,
		scale:   float64(srcW) / float64(dstW),
		support: f.Support,
		left:    make([]int32, dstW),
		off:     make([]int32, dstW),
		cnt:     make([]uint16, dstW),
	}
	supp := f.Support * ct.scale

	// first pass: total taps
	total := 0
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
	ct.wQ15 = make([]int16, total)

	// second pass: fill meta + Q15 weights normalized to sum==32768
	const ONE = 1<<15 - 1 // 32768
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

		// compute float weights + sum
		sum := 0.0
		// keep track of argmax for residual fixup
		maxIdx := 0
		maxAbs := 0.0
		for i := 0; i < taps; i++ {
			si := left + i
			dx := (float64(si) + 0.5 - center) / ct.scale
			ww := f.Kernel(dx)
			sum += ww
			if abs := math.Abs(ww); abs > maxAbs {
				maxAbs = abs
				maxIdx = i
			}
			// temp store as float in a scratch area? we can reuse ct.wQ15 then overwrite
			// but simpler: store in a small local slice
		}
		if sum == 0 {
			// NN fallback
			sx := int(center)
			if sx < 0 {
				sx = 0
			} else if sx >= srcW {
				sx = srcW - 1
			}
			ct.left[x] = int32(sx)
			ct.off[x] = int32(wpos)
			ct.cnt[x] = 1
			ct.wQ15[wpos] = int16(ONE)
			wpos += 1
			continue
		}
		// second pass for this pixel: quantize to Q15 and normalize exactly
		// Recompute weights to avoid storing a temp slice (keeps allocs at zero)
		sumQ := 0
		for i := 0; i < taps; i++ {
			si := left + i
			dx := (float64(si) + 0.5 - center) / ct.scale
			ww := f.Kernel(dx) / sum
			q := int(math.Round(ww * ONE))
			// clamp to int16 range
			if q > 32767 {
				q = 32767
			}
			if q < -32768 {
				q = -32768
			}
			ct.wQ15[wpos+i] = int16(q)
			sumQ += q
		}
		// force exact normalization by fixing the largest-magnitude tap
		if sumQ != ONE {
			i := maxIdx
			idx := wpos + i
			fix := ONE - sumQ
			val := int(ct.wQ15[idx]) + fix
			if val > 32767 {
				val = 32767
			}
			if val < -32768 {
				val = -32768
			}
			ct.wQ15[idx] = int16(val)
		}
		wpos += taps
	}
	return ct
}

var (
	coeffCacheQ15Mu sync.Mutex
	coeffCacheQ15   struct {
		srcW, dstW int
		support    float64
		ct         *coeffTableQ15
	}
)

func getCoeffTableQ15(srcW, dstW int, f Filter) *coeffTableQ15 {
	coeffCacheQ15Mu.Lock()
	c := coeffCacheQ15
	if c.ct != nil && c.srcW == srcW && c.dstW == dstW && c.support == f.Support {
		ct := c.ct
		coeffCacheQ15Mu.Unlock()
		return ct
	}
	coeffCacheQ15Mu.Unlock()

	ct := buildCoeffTableFlatQ15(srcW, dstW, f)

	coeffCacheQ15Mu.Lock()
	coeffCacheQ15 = struct {
		srcW, dstW int
		support    float64
		ct         *coeffTableQ15
	}{srcW, dstW, f.Support, ct}
	coeffCacheQ15Mu.Unlock()
	return ct
}

// Uses int32 accumulator (safe: 255*32768*taps; taps typically <= ~100).
func resampleGrayRowIntoQ15(dst []byte, dstW int, src []byte, srcW int, f Filter) {
	if dstW <= 0 || srcW <= 0 {
		return
	}
	ct := getCoeffTableQ15(srcW, dstW, f)
	const SHIFT = 15
	const ROUND = 1 << (SHIFT - 1)
	for x := 0; x < dstW; x++ {
		left := int(ct.left[x])
		off := int(ct.off[x])
		n := int(ct.cnt[x])

		var acc int32
		wi := off
		si := left
		for i := 0; i < n; i++ {
			acc += int32(src[si]) * int32(ct.wQ15[wi])
			si++
			wi++
		}
		v := int32((acc + ROUND) >> SHIFT)
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		dst[x] = byte(v)
	}
}

func resampleRGBtoRGBAIntoQ15(dst []byte, dstW int, src []byte, srcW int, f Filter, alpha byte) {
	if dstW <= 0 || srcW <= 0 {
		return
	}
	ct := getCoeffTableQ15(srcW, dstW, f)
	const SHIFT = 15
	const ROUND = 1 << (SHIFT - 1)
	for x := 0; x < dstW; x++ {
		left := int(ct.left[x])
		off := int(ct.off[x])
		n := int(ct.cnt[x])

		var r, g, b int32
		si := left * 3
		wi := off
		for i := 0; i < n; i++ {
			w := int32(ct.wQ15[wi])
			r += int32(src[si+0]) * w
			g += int32(src[si+1]) * w
			b += int32(src[si+2]) * w
			si += 3
			wi++
		}
		R := (r + ROUND) >> SHIFT
		G := (g + ROUND) >> SHIFT
		B := (b + ROUND) >> SHIFT
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
		i := x * 4
		dst[i+0] = byte(R)
		dst[i+1] = byte(G)
		dst[i+2] = byte(B)
		dst[i+3] = alpha
	}
}

func resampleRGBAPremulIntoQ15(dst []byte, dstW int, src []byte, srcW int, f Filter) {
	if dstW <= 0 || srcW <= 0 {
		return
	}
	ct := getCoeffTableQ15(srcW, dstW, f)
	for x := 0; x < dstW; x++ {
		left := int(ct.left[x])
		off := int(ct.off[x])
		n := int(ct.cnt[x])

		var ar, ag, ab, aa int64
		si := left * 4
		wi := off
		for i := 0; i < n; i++ {
			w := int64(ct.wQ15[wi])     // Q15
			a := int64(src[si+3])       // 0..255
			aa += a * w                 // Q15*alpha
			aw := a * w                 // Q15*alpha
			ar += int64(src[si+0]) * aw // Q15*(r*a)
			ag += int64(src[si+1]) * aw
			ab += int64(src[si+2]) * aw
			si += 4
			wi++
		}
		var R, G, B, A int64
		A = (aa + (1 << 14)) >> 15 // scale back to 0..255 with rounding
		if aa > 0 {
			// R = (ar/aa) with rounding
			R = (ar + aa/2) / aa
			G = (ag + aa/2) / aa
			B = (ab + aa/2) / aa
		} else {
			R, G, B = 0, 0, 0
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
		if A < 0 {
			A = 0
		} else if A > 255 {
			A = 255
		}
		i := x * 4
		dst[i+0] = byte(R)
		dst[i+1] = byte(G)
		dst[i+2] = byte(B)
		dst[i+3] = byte(A)
	}
}

func resampleGrayToRGBAIntoQ15(dst []byte, dstW int, src []byte, srcW int, f Filter, alpha byte) {
	if dstW <= 0 || srcW <= 0 {
		return
	}
	ct := getCoeffTableQ15(srcW, dstW, f)
	const SHIFT = 15
	const ROUND = 1 << (SHIFT - 1)
	for x := 0; x < dstW; x++ {
		left := int(ct.left[x])
		off := int(ct.off[x])
		n := int(ct.cnt[x])

		var v int32
		wi := off
		si := left
		for i := 0; i < n; i++ {
			v += int32(src[si]) * int32(ct.wQ15[wi])
			si++
			wi++
		}
		V := (v + ROUND) >> SHIFT
		if V < 0 {
			V = 0
		} else if V > 255 {
			V = 255
		}
		iv := byte(V)
		di := x * 4
		dst[di+0] = iv
		dst[di+1] = iv
		dst[di+2] = iv
		dst[di+3] = alpha
	}
}
