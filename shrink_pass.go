package pngscaled

import (
	"image"
	"image/color"
)

// shrinkRowG8 resamples a Gray-8 source row into the destination image.
// It handles both the pure-gray and gray+tRNS cases.
// Returns the updated pixOffset.
func (d *decoder) shrinkRowG8(cdat []byte, nrgba *image.NRGBA, gray *image.Gray, pixOffset, width int) int {
	dstW := d.targetWidth
	if d.useTransparent {
		t := d.transparent[1]
		row := nrgba.Pix[pixOffset : pixOffset+nrgba.Stride]
		resampleGrayTRNSPremulIntoNRGBAQ15(row[:dstW*4], dstW, cdat, width, d.filter, t)
		for k := dstW * 4; k < nrgba.Rect.Dx()*4; k++ {
			row[k] = 0
		}
		return pixOffset + nrgba.Stride
	}
	row := gray.Pix[pixOffset : pixOffset+gray.Stride]
	resampleGrayRowIntoQ15(row[:dstW], dstW, cdat, width, d.filter)
	for k := dstW; k < gray.Rect.Dx(); k++ {
		row[k] = 0
	}
	return pixOffset + gray.Stride
}

// shrinkRowGA8 resamples a GrayAlpha-8 source row into the destination NRGBA image.
// Returns the updated pixOffset.
func (d *decoder) shrinkRowGA8(cdat []byte, nrgba *image.NRGBA, pixOffset, width int) int {
	dstW := d.targetWidth
	row := nrgba.Pix[pixOffset : pixOffset+nrgba.Stride]
	resampleGrayAlphaPremulIntoNRGBAQ15(row[:dstW*4], dstW, cdat, width, d.filter)
	for k := dstW * 4; k < nrgba.Rect.Dx()*4; k++ {
		row[k] = 0
	}
	return pixOffset + nrgba.Stride
}

// shrinkRowTC8 resamples an RGB-8 source row into the destination image.
// It handles both the plain-RGB and RGB+tRNS cases.
// Returns the updated pixOffset.
func (d *decoder) shrinkRowTC8(cdat []byte, nrgba *image.NRGBA, rgba *image.RGBA, pixOffset, width int) int {
	dstW := d.targetWidth
	if d.useTransparent {
		tr, tg, tb := d.transparent[1], d.transparent[3], d.transparent[5]
		row := nrgba.Pix[pixOffset : pixOffset+nrgba.Stride]
		resampleRGBTRNSPremulIntoNRGBAQ15(row[:dstW*4], dstW, cdat, width, d.filter, tr, tg, tb)
		return pixOffset + nrgba.Stride
	}
	row := rgba.Pix[pixOffset : pixOffset+rgba.Stride]
	resampleRGBtoRGBAIntoQ15(row[:dstW*4], dstW, cdat, width, d.filter, 0xff)
	for k := dstW * 4; k < rgba.Rect.Dx()*4; k++ {
		row[k] = 0
	}
	return pixOffset + rgba.Stride
}

// shrinkRowTCA8 resamples an RGBA-8 source row into the destination NRGBA image.
// Returns the updated pixOffset.
func (d *decoder) shrinkRowTCA8(cdat []byte, nrgba *image.NRGBA, pixOffset, width int) int {
	dstW := d.targetWidth
	row := nrgba.Pix[pixOffset : pixOffset+nrgba.Stride]
	resampleRGBAPremulIntoQ15(row[:dstW*4], dstW, cdat, width, d.filter)
	for k := dstW * 4; k < nrgba.Rect.Dx()*4; k++ {
		row[k] = 0
	}
	return pixOffset + nrgba.Stride
}

// shrinkRowG16 resamples a Gray-16 source row into the destination image.
// Returns the updated pixOffset.
func (d *decoder) shrinkRowG16(cdat []byte, nrgba64 *image.NRGBA64, gray16 *image.Gray16, pixOffset, width int) int {
	dstW := d.targetWidth
	if d.useTransparent {
		ty := uint16(d.transparent[0])<<8 | uint16(d.transparent[1])
		row := nrgba64.Pix[pixOffset : pixOffset+nrgba64.Stride]
		resampleGray16TRNSPremulIntoNRGBA64Q15(row[:dstW*8], dstW, cdat, width, d.filter, ty)
		return pixOffset + nrgba64.Stride
	}
	row := gray16.Pix[pixOffset : pixOffset+gray16.Stride]
	resampleGray16RowIntoQ15(row[:dstW*2], dstW, cdat, width, d.filter)
	return pixOffset + gray16.Stride
}

// shrinkRowGA16 resamples a GrayAlpha-16 source row into the destination NRGBA64 image.
// Returns the updated pixOffset.
func (d *decoder) shrinkRowGA16(cdat []byte, nrgba64 *image.NRGBA64, pixOffset, width int) int {
	dstW := d.targetWidth
	row := nrgba64.Pix[pixOffset : pixOffset+nrgba64.Stride]
	resampleGrayAlpha16PremulIntoNRGBA64Q15(row[:dstW*8], dstW, cdat, width, d.filter)
	return pixOffset + nrgba64.Stride
}

// shrinkRowTC16 resamples an RGB-16 source row into the destination image.
// Returns the updated pixOffset.
func (d *decoder) shrinkRowTC16(cdat []byte, nrgba64 *image.NRGBA64, rgba64 *image.RGBA64, pixOffset, width int) int {
	dstW := d.targetWidth
	if d.useTransparent {
		tr := uint16(d.transparent[0])<<8 | uint16(d.transparent[1])
		tg := uint16(d.transparent[2])<<8 | uint16(d.transparent[3])
		tb := uint16(d.transparent[4])<<8 | uint16(d.transparent[5])
		row := nrgba64.Pix[pixOffset : pixOffset+nrgba64.Stride]
		resampleRGB16TRNSPremulIntoNRGBA64Q15(row[:dstW*8], dstW, cdat, width, d.filter, tr, tg, tb)
		return pixOffset + nrgba64.Stride
	}
	row := rgba64.Pix[pixOffset : pixOffset+rgba64.Stride]
	resampleRGB16toRGBA64BytesIntoQ15(row[:dstW*8], dstW, cdat, width, d.filter, 0xffff)
	return pixOffset + rgba64.Stride
}

// shrinkRowTCA16 resamples an RGBA-16 source row into the destination NRGBA64 image.
// Returns the updated pixOffset.
func (d *decoder) shrinkRowTCA16(cdat []byte, nrgba64 *image.NRGBA64, pixOffset, width int) int {
	dstW := d.targetWidth
	row := nrgba64.Pix[pixOffset : pixOffset+nrgba64.Stride]
	resampleRGBA16PremulIntoNRGBA64Q15(row[:dstW*8], dstW, cdat, width, d.filter)
	return pixOffset + nrgba64.Stride
}

// palEntryToRGBA8 extracts straight (non-premultiplied) RGBA 8-bit values from
// a palette entry. Palette entries are always color.RGBA or color.NRGBA, so we
// read their fields directly instead of calling .RGBA() which returns premultiplied
// 16-bit values.
func palEntryToRGBA8(c color.Color) (r, g, b, a byte) {
	switch cv := c.(type) {
	case color.RGBA:
		return cv.R, cv.G, cv.B, cv.A
	case color.NRGBA:
		return cv.R, cv.G, cv.B, cv.A
	default:
		// Generic fallback: un-premultiply from .RGBA().
		rr, gg, bb, aa := c.RGBA()
		a = byte(aa >> 8)
		if aa > 0 {
			r = byte(rr * 0xff / aa)
			g = byte(gg * 0xff / aa)
			b = byte(bb * 0xff / aa)
		}
		return
	}
}

// shrinkRowP expands paletted source row indices to RGBA and resamples into the destination NRGBA image.
// Returns the updated pixOffset.
func (d *decoder) shrinkRowP(cdat []byte, depth int, palette color.Palette, rowBuf []byte, nrgba *image.NRGBA, pixOffset, width int) int {
	pos := 0
	switch depth {
	case 1:
		for x := 0; x < width; x += 8 {
			b := cdat[x/8]
			for x2 := 0; x2 < 8 && x+x2 < width; x2++ {
				idx := b >> 7
				var c color.Color
				if int(idx) < len(palette) {
					c = palette[idx]
				} else {
					c = color.RGBA{0, 0, 0, 0xff}
				}
				r, g, bl, a := palEntryToRGBA8(c)
				rowBuf[pos+0] = r
				rowBuf[pos+1] = g
				rowBuf[pos+2] = bl
				rowBuf[pos+3] = a
				pos += 4
				b <<= 1
			}
		}
	case 2:
		for x := 0; x < width; x += 4 {
			b := cdat[x/4]
			for x2 := 0; x2 < 4 && x+x2 < width; x2++ {
				idx := b >> 6
				var c color.Color
				if int(idx) < len(palette) {
					c = palette[idx]
				} else {
					c = color.RGBA{0, 0, 0, 0xff}
				}
				r, g, bl, a := palEntryToRGBA8(c)
				rowBuf[pos+0] = r
				rowBuf[pos+1] = g
				rowBuf[pos+2] = bl
				rowBuf[pos+3] = a
				pos += 4
				b <<= 2
			}
		}
	case 4:
		for x := 0; x < width; x += 2 {
			b := cdat[x/2]
			for x2 := 0; x2 < 2 && x+x2 < width; x2++ {
				idx := b >> 4
				var c color.Color
				if int(idx) < len(palette) {
					c = palette[idx]
				} else {
					c = color.RGBA{0, 0, 0, 0xff}
				}
				r, g, bl, a := palEntryToRGBA8(c)
				rowBuf[pos+0] = r
				rowBuf[pos+1] = g
				rowBuf[pos+2] = bl
				rowBuf[pos+3] = a
				pos += 4
				b <<= 4
			}
		}
	case 8:
		for x := 0; x < width; x++ {
			idx := cdat[x]
			var c color.Color
			if int(idx) < len(palette) {
				c = palette[idx]
			} else {
				c = color.RGBA{0, 0, 0, 0xff}
			}
			r, g, bl, a := palEntryToRGBA8(c)
			rowBuf[pos+0] = r
			rowBuf[pos+1] = g
			rowBuf[pos+2] = bl
			rowBuf[pos+3] = a
			pos += 4
		}
	}
	dstW := d.targetWidth
	row := nrgba.Pix[pixOffset : pixOffset+nrgba.Stride]
	resampleRGBAPremulIntoQ15(row[:dstW*4], dstW, rowBuf[:width*4], width, d.filter)
	return pixOffset + nrgba.Stride
}
