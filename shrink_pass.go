package pngscaled

import "image"

// shrinkRowG8 resamples a Gray-8 source row into the destination image.
// It handles both the pure-gray and gray+tRNS cases.
// Returns the updated pixOffset.
func (d *decoder) shrinkRowG8(cdat []byte, nrgba *image.NRGBA, gray *image.Gray, pixOffset, width int) int {
	dstW := d.targetWidth
	if d.useTransparent {
		t := d.transparent[1]
		row := nrgba.Pix[pixOffset : pixOffset+nrgba.Stride]
		resampleGrayToRGBAIntoQ15(row[:dstW*4], dstW, cdat, width, d.filter, 0xff)
		for x := 0; x < dstW; x++ {
			i := x * 4
			if row[i+0] == t {
				row[i+3] = 0x00
			}
		}
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
		resampleRGBtoRGBAIntoQ15(row[:dstW*4], dstW, cdat, width, d.filter, 0xff)
		for x := 0; x < dstW; x++ {
			i := x * 4
			if row[i+0] == tr && row[i+1] == tg && row[i+2] == tb {
				row[i+3] = 0x00
			}
		}
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
