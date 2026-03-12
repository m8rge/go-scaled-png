package pngscaled

import "image"

type rowScaleScratch struct {
	palettedRGBA []byte
	grayExpanded []byte
}

func (d *decoder) shouldScaleRow(width int) bool {
	return d.targetWidth > 0 && d.targetWidth < width && d.interlace == itNone
}

func (d *decoder) newRowScaleScratch(width int) rowScaleScratch {
	if !d.shouldScaleRow(width) {
		return rowScaleScratch{}
	}

	s := rowScaleScratch{}
	if cbPaletted(d.cb) {
		s.palettedRGBA = make([]byte, width*4)
	}
	if d.cb == cbG1 || d.cb == cbG2 || d.cb == cbG4 {
		s.grayExpanded = make([]byte, width)
	}
	return s
}

func (d *decoder) scaleRowIfNeeded(
	cdat []byte,
	width int,
	pixOffset *int,
	scratch *rowScaleScratch,
	gray *image.Gray,
	rgba *image.RGBA,
	nrgba *image.NRGBA,
	gray16 *image.Gray16,
	rgba64 *image.RGBA64,
	nrgba64 *image.NRGBA64,
) bool {
	if !d.shouldScaleRow(width) {
		return false
	}

	switch d.cb {
	case cbG1:
		expandGray1(scratch.grayExpanded, cdat, width)
		*pixOffset = d.shrinkRowG8(scratch.grayExpanded, nrgba, gray, *pixOffset, width)
	case cbG2:
		expandGray2(scratch.grayExpanded, cdat, width)
		*pixOffset = d.shrinkRowG8(scratch.grayExpanded, nrgba, gray, *pixOffset, width)
	case cbG4:
		expandGray4(scratch.grayExpanded, cdat, width)
		*pixOffset = d.shrinkRowG8(scratch.grayExpanded, nrgba, gray, *pixOffset, width)
	case cbG8:
		*pixOffset = d.shrinkRowG8(cdat, nrgba, gray, *pixOffset, width)
	case cbGA8:
		*pixOffset = d.shrinkRowGA8(cdat, nrgba, *pixOffset, width)
	case cbTC8:
		*pixOffset = d.shrinkRowTC8(cdat, nrgba, rgba, *pixOffset, width)
	case cbP1, cbP2, cbP4, cbP8:
		*pixOffset = d.shrinkRowP(cdat, d.depth, d.palette, scratch.palettedRGBA, nrgba, *pixOffset, width)
	case cbTCA8:
		*pixOffset = d.shrinkRowTCA8(cdat, nrgba, *pixOffset, width)
	case cbG16:
		*pixOffset = d.shrinkRowG16(cdat, nrgba64, gray16, *pixOffset, width)
	case cbGA16:
		*pixOffset = d.shrinkRowGA16(cdat, nrgba64, *pixOffset, width)
	case cbTC16:
		*pixOffset = d.shrinkRowTC16(cdat, nrgba64, rgba64, *pixOffset, width)
	case cbTCA16:
		*pixOffset = d.shrinkRowTCA16(cdat, nrgba64, *pixOffset, width)
	default:
		return false
	}

	return true
}

func expandGray1(dst, src []byte, width int) {
	for x := 0; x < width; x += 8 {
		b := src[x/8]
		for x2 := 0; x2 < 8 && x+x2 < width; x2++ {
			dst[x+x2] = (b >> 7) * 0xff
			b <<= 1
		}
	}
}

func expandGray2(dst, src []byte, width int) {
	for x := 0; x < width; x += 4 {
		b := src[x/4]
		for x2 := 0; x2 < 4 && x+x2 < width; x2++ {
			dst[x+x2] = (b >> 6) * 0x55
			b <<= 2
		}
	}
}

func expandGray4(dst, src []byte, width int) {
	for x := 0; x < width; x += 2 {
		b := src[x/2]
		for x2 := 0; x2 < 2 && x+x2 < width; x2++ {
			dst[x+x2] = (b >> 4) * 0x11
			b <<= 4
		}
	}
}
