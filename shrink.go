package pngscaled

import (
	"image"
	"math"
	"sync"
)

// Q15 fixed-point weights table: one build per (srcW, dstW, filter.Support).
// We store weights in int16 where 1.0 == 32768 (1<<15). Per-pixel weights sum to 32768.
type coeffTableQ15 struct {
	scale float64
	left  []int32  // len == dstW
	off   []int32  // len == dstW, offset into wQ15
	cnt   []uint16 // len == dstW, number of taps
	wQ15  []int16  // flat weights, all pixels concatenated
}

func buildCoeffTableFlatQ15(srcW, dstW int, f ResampleFilter) *coeffTableQ15 {
	ct := &coeffTableQ15{
		scale: float64(srcW) / float64(dstW),
		left:  make([]int32, dstW),
		off:   make([]int32, dstW),
		cnt:   make([]uint16, dstW),
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

// Build (or reuse) vertical Q15 coefficients.
// Reuses coeffTableQ15 shape: left/off/cnt are sized for dstH, wQ15 is flat.
func getCoeffTableQ15Y(srcH, dstH int, f ResampleFilter) *coeffTableQ15 {
	if srcH <= 0 || dstH <= 0 {
		return nil
	}
	coeffCacheQ15YMu.Lock()
	cached := coeffCacheQ15Y.ct != nil &&
		coeffCacheQ15Y.srcH == srcH &&
		coeffCacheQ15Y.dstH == dstH &&
		coeffCacheQ15Y.support == f.Support
	var ct *coeffTableQ15
	if cached {
		ct = coeffCacheQ15Y.ct
		coeffCacheQ15YMu.Unlock()
		return ct
	}
	coeffCacheQ15YMu.Unlock()

	scale := float64(srcH) / float64(dstH) // >1 for downscale
	sup := f.Support

	left := make([]int32, dstH)
	off := make([]int32, dstH)
	cnt := make([]uint16, dstH)
	wQ15 := make([]int16, 0, dstH*int(2*math.Ceil(sup*scale)+3)) // rough cap

	wOffset := 0
	for y := 0; y < dstH; y++ {
		// Map dest center to source space (same convention as X path)
		yc := (float64(y)+0.5)*scale - 0.5

		start := int(math.Ceil(yc - sup*scale))
		end := int(math.Floor(yc + sup*scale))
		if start < 0 {
			start = 0
		}
		if end > srcH-1 {
			end = srcH - 1
		}
		if end < start {
			end = start
		}
		n := end - start + 1

		// Float normalization
		sum := 0.0
		for s := start; s <= end; s++ {
			t := (yc - float64(s)) / scale
			sum += f.Kernel(t)
		}
		if sum == 0 {
			sum = 1
		}

		sumQ := int32(0)
		rowOff := wOffset
		for s := start; s <= end; s++ {
			t := (yc - float64(s)) / scale
			w := f.Kernel(t) / sum
			q := int32(math.Round(w * 32768.0))
			if q < -32768 {
				q = -32768
			}
			if q > 32767 {
				q = 32767
			}
			wQ15 = append(wQ15, int16(q))
			wOffset++
			sumQ += q
		}
		// Fix rounding on the last tap so Σw == 32768
		if n > 0 && sumQ != 32768 {
			wQ15[wOffset-1] = int16(int32(wQ15[wOffset-1]) + (32768 - sumQ))
		}

		left[y] = int32(start)
		off[y] = int32(rowOff)
		cnt[y] = uint16(n)
	}

	ct = &coeffTableQ15{
		scale: scale,
		left:  left,
		off:   off,
		cnt:   cnt,
		wQ15:  wQ15,
	}

	coeffCacheQ15YMu.Lock()
	coeffCacheQ15Y.srcH = srcH
	coeffCacheQ15Y.dstH = dstH
	coeffCacheQ15Y.support = sup
	coeffCacheQ15Y.ct = ct
	coeffCacheQ15YMu.Unlock()
	return ct
}

var (
	coeffCacheQ15Mu sync.Mutex
	coeffCacheQ15   struct {
		srcW, dstW int
		support    float64
		ct         *coeffTableQ15
	}
	coeffCacheQ15YMu sync.Mutex
	coeffCacheQ15Y   struct {
		srcH, dstH int
		support    float64
		ct         *coeffTableQ15 // left/off/cnt sized to dstH; wQ15 concatenated
	}
)

func clamp8(v int32) byte {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(v)
}

func getCoeffTableQ15(srcW, dstW int, f ResampleFilter) *coeffTableQ15 {
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
func resampleGrayRowIntoQ15(dst []byte, dstW int, src []byte, srcW int, f ResampleFilter) {
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
		dst[x] = clamp8((acc + ROUND) >> SHIFT)
	}
}

func resampleRGBtoRGBAIntoQ15(dst []byte, dstW int, src []byte, srcW int, f ResampleFilter, alpha byte) {
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
		i := x * 4
		dst[i+0] = clamp8((r + ROUND) >> SHIFT)
		dst[i+1] = clamp8((g + ROUND) >> SHIFT)
		dst[i+2] = clamp8((b + ROUND) >> SHIFT)
		dst[i+3] = alpha
	}
}

func resampleRGBAPremulIntoQ15(dst []byte, dstW int, src []byte, srcW int, f ResampleFilter) {
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
		i := x * 4
		dst[i+0] = clamp8(int32(R))
		dst[i+1] = clamp8(int32(G))
		dst[i+2] = clamp8(int32(B))
		dst[i+3] = clamp8(int32(A))
	}
}

func resampleGrayToRGBAIntoQ15(dst []byte, dstW int, src []byte, srcW int, f ResampleFilter, alpha byte) {
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
		iv := clamp8((v + ROUND) >> SHIFT)
		di := x * 4
		dst[di+0] = iv
		dst[di+1] = iv
		dst[di+2] = iv
		dst[di+3] = alpha
	}
}

func verticalRGBAColumnsQ15(pix []byte, stride, width, dstH int, ct *coeffTableQ15) {
	if ct == nil || dstH <= 0 {
		return
	}
	buf := make([]byte, 4*dstH)
	for x := 0; x < width; x++ {
		colOffset := 4 * x
		for yd := 0; yd < dstH; yd++ {
			start := int(ct.left[yd])
			n := int(ct.cnt[yd])
			wi := int(ct.off[yd])

			var r, g, b, a int32
			for k := 0; k < n; k++ {
				q := int32(ct.wQ15[wi+k])
				p := pix[(start+k)*stride+colOffset:]
				r += int32(p[0]) * q
				g += int32(p[1]) * q
				b += int32(p[2]) * q
				a += int32(p[3]) * q
			}
			i := 4 * yd
			buf[i+0] = clamp8((r + 16384) >> 15)
			buf[i+1] = clamp8((g + 16384) >> 15)
			buf[i+2] = clamp8((b + 16384) >> 15)
			buf[i+3] = clamp8((a + 16384) >> 15)
		}
		for yd := 0; yd < dstH; yd++ {
			copy(pix[yd*stride+colOffset:], buf[4*yd:4*yd+4])
		}
	}
}

// RGBA (straight alpha)
func VerticalRGBAInPlaceQ15(img *image.RGBA, dstH int, f ResampleFilter) {
	if img == nil || dstH <= 0 || dstH >= img.Rect.Dy() {
		return
	}
	srcH, w, stride := img.Rect.Dy(), img.Rect.Dx(), img.Stride
	ct := getCoeffTableQ15Y(srcH, dstH, f)
	verticalRGBAColumnsQ15(img.Pix, stride, w, dstH, ct)
	img.Rect.Max.Y = img.Rect.Min.Y + dstH
}

// NRGBA (premultiplied alpha preserved)
func VerticalNRGBAInPlaceQ15(img *image.NRGBA, dstH int, f ResampleFilter) {
	if img == nil || dstH <= 0 || dstH >= img.Rect.Dy() {
		return
	}
	srcH, w, stride := img.Rect.Dy(), img.Rect.Dx(), img.Stride
	ct := getCoeffTableQ15Y(srcH, dstH, f)
	verticalRGBAColumnsQ15(img.Pix, stride, w, dstH, ct)
	img.Rect.Max.Y = img.Rect.Min.Y + dstH
}

// Gray (8-bit)
func VerticalGrayInPlaceQ15(img *image.Gray, dstH int, f ResampleFilter) {
	if img == nil || dstH <= 0 || dstH >= img.Rect.Dy() {
		return
	}
	srcH, w, stride := img.Rect.Dy(), img.Rect.Dx(), img.Stride
	ct := getCoeffTableQ15Y(srcH, dstH, f)
	buf := make([]byte, dstH)

	for x := 0; x < w; x++ {
		for yd := 0; yd < dstH; yd++ {
			start := int(ct.left[yd])
			n := int(ct.cnt[yd])
			wi := int(ct.off[yd])

			var acc int32
			for k := 0; k < n; k++ {
				q := int32(ct.wQ15[wi+k])
				acc += int32(img.Pix[(start+k)*stride+x]) * q
			}
			buf[yd] = clamp8((acc + 16384) >> 15)
		}
		for yd := 0; yd < dstH; yd++ {
			img.Pix[yd*stride+x] = buf[yd]
		}
	}
	img.Rect.Max.Y = img.Rect.Min.Y + dstH
}
