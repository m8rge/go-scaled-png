package pngscaled

import "math"

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

// resampleRowInto writes a horizontally resampled row into dst (no allocations).
//   - dst must have room for dstW * chansOut bytes (we usually pass left part of Pix row).
//   - src is a single de-filtered row: srcW * chansIn bytes.
//   - If chansOut > chansIn (e.g. RGB->RGBA), alphaConst is used to fill the last channel
//     when fillAlphaConst == true.
//
// Edge handling: clamp-to-edge (same as common imaging libraries).
func resampleRowInto(
	dst []byte, dstW int, src []byte, srcW, chansIn, chansOut int,
	filter Filter, fillAlphaConst bool, alphaConst byte,
) {
	if dstW <= 0 || srcW <= 0 {
		return
	}
	scale := float64(srcW) / float64(dstW)
	support := filter.Support * scale
	for x := 0; x < dstW; x++ {
		center := (float64(x) + 0.5) * scale
		left := int(math.Ceil(center - support))
		right := int(math.Floor(center + support))
		for c := 0; c < chansOut; c++ {
			if c == chansOut-1 && fillAlphaConst && chansOut > chansIn {
				dst[x*chansOut+c] = alphaConst
				continue
			}
			srcC := c
			if c >= chansIn {
				srcC = chansIn - 1
			} // safety
			var sum float64
			for i := left; i <= right; i++ {
				ii := i
				if ii < 0 {
					ii = 0
				} else if ii >= srcW {
					ii = srcW - 1
				}
				// Distance in source pixel domain, normalized by scale:
				w := filter.Kernel((float64(ii) + 0.5 - center) / scale)
				sum += float64(src[ii*chansIn+srcC]) * w
			}
			if sum < 0 {
				sum = 0
			} else if sum > 255 {
				sum = 255
			}
			dst[x*chansOut+c] = byte(sum + 0.5)
		}
	}
}

// Filter and UseImagingFilter you already have.

// resampleRowIntoNormalized: normalized weights (sum to 1) for non-premultiplied data.
// chansIn/out as before. No allocations.
func resampleRowIntoNormalized(
	dst []byte, dstW int, src []byte, srcW, chansIn, chansOut int,
	filter Filter, fillAlphaConst bool, alphaConst byte,
) {
	if dstW <= 0 || srcW <= 0 {
		return
	}
	scale := float64(srcW) / float64(dstW)
	support := filter.Support * scale
	for x := 0; x < dstW; x++ {
		center := (float64(x) + 0.5) * scale
		left := int(math.Ceil(center - support))
		right := int(math.Floor(center + support))

		var sumW float64
		ws := make([]float64, right-left+1) // tiny, per-pixel; if you want zero alloc, reuse a small stack buffer
		idx := 0
		for i := left; i <= right; i++ {
			ii := i
			if ii < 0 {
				ii = 0
			} else if ii >= srcW {
				ii = srcW - 1
			}
			w := filter.Kernel((float64(ii) + 0.5 - center) / scale)
			ws[idx] = w
			sumW += w
			idx++
		}
		if sumW == 0 {
			// Fallback to nearest neighbor.
			sx := int(center)
			if sx < 0 {
				sx = 0
			} else if sx >= srcW {
				sx = srcW - 1
			}
			copy(dst[x*chansOut:x*chansOut+min(chansOut, chansIn)], src[sx*chansIn:sx*chansIn+chansIn])
			if fillAlphaConst && chansOut > chansIn {
				dst[x*chansOut+chansOut-1] = alphaConst
			}
			continue
		}
		inv := 1.0 / sumW

		for c := 0; c < chansOut; c++ {
			if c == chansOut-1 && fillAlphaConst && chansOut > chansIn {
				dst[x*chansOut+c] = alphaConst
				continue
			}
			srcC := c
			if srcC >= chansIn {
				srcC = chansIn - 1
			}
			var acc float64
			for j := 0; j < len(ws); j++ {
				i := left + j
				if i < 0 {
					i = 0
				} else if i >= srcW {
					i = srcW - 1
				}
				acc += float64(src[i*chansIn+srcC]) * (ws[j] * inv)
			}
			if acc < 0 {
				acc = 0
			} else if acc > 255 {
				acc = 255
			}
			dst[x*chansOut+c] = byte(acc + 0.5)
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// resampleRowRGBAIntoPremulNormalized: proper RGBA resampling in premultiplied-alpha space.
// src and dst are RGBA (4 chans), no allocations, normalized weights.
func resampleRowRGBAIntoPremulNormalized(dst []byte, dstW int, src []byte, srcW int, filter Filter) {
	if dstW <= 0 || srcW <= 0 {
		return
	}
	const chans = 4
	scale := float64(srcW) / float64(dstW)
	support := filter.Support * scale
	for x := 0; x < dstW; x++ {
		center := (float64(x) + 0.5) * scale
		left := int(math.Ceil(center - support))
		right := int(math.Floor(center + support))

		var sumW float64
		ws := make([]float64, right-left+1)
		idx := 0
		for i := left; i <= right; i++ {
			ii := i
			if ii < 0 {
				ii = 0
			} else if ii >= srcW {
				ii = srcW - 1
			}
			w := filter.Kernel((float64(ii) + 0.5 - center) / scale)
			ws[idx] = w
			sumW += w
			idx++
		}
		if sumW == 0 {
			// nearest
			sx := int(center)
			if sx < 0 {
				sx = 0
			} else if sx >= srcW {
				sx = srcW - 1
			}
			copy(dst[x*chans:x*chans+chans], src[sx*chans:sx*chans+chans])
			continue
		}
		inv := 1.0 / sumW

		var accR, accG, accB, accA float64
		for j := 0; j < len(ws); j++ {
			i := left + j
			if i < 0 {
				i = 0
			} else if i >= srcW {
				i = srcW - 1
			}
			w := ws[j] * inv
			si := i * chans
			a := float64(src[si+3])
			accA += a * w
			// premultiply
			aw := a * w
			accR += float64(src[si+0]) * aw
			accG += float64(src[si+1]) * aw
			accB += float64(src[si+2]) * aw
		}
		// unpremultiply
		var R, G, B, A float64
		A = accA
		if A > 0 {
			R = accR / A
			G = accG / A
			B = accB / A
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
		i := x * chans
		dst[i+0] = byte(R + 0.5)
		dst[i+1] = byte(G + 0.5)
		dst[i+2] = byte(B + 0.5)
		dst[i+3] = byte(A + 0.5)
	}
}
