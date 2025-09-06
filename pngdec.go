// Package pngdec provides a tiny PNG decoder with a per-line callback,
// inspired by bitbank2/PNGdec but implemented in pure Go.
//
// MVP features:
//   - 8-bit depth only (grayscale, RGB, RGBA, indexed, gray+alpha)
//   - Non-interlaced images
//   - Per-scanline callback delivering RGBA bytes
//   - Returns image.Image (RGBA) for easy benchmarking vs image/png
//
// Planned (not yet implemented):
//   - Adam7 interlace
//   - Bit depths 1/2/4/16
//   - Ancillary chunks (gamma, cHRM, sRGB, etc)
//
// This is an independent clean-room implementation using the PNG spec.
package pngscaled

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
)

// LineCallback is invoked for each decoded scanline in RGBA format.
// The slice is width*4 bytes and is only valid for the duration of the call.
//
// y is the 0-based row index.
//
// If nil, callbacks are skipped.
type LineCallback func(y int, rgba []byte)

// Decode reads a PNG from r, optionally issuing a per-line callback, and returns an RGBA image.
// MVP: supports only 8-bit per channel, non-interlaced PNGs. Other cases return an error.
func Decode(r io.Reader, cb LineCallback) (*image.RGBA, error) {
	pr := &pngReader{r: r}
	if err := pr.readSignature(); err != nil {
		return nil, err
	}

	var ihdr ihdrData
	var palette []color.NRGBA // PLTE entries
	var trns *tRNSData        // transparency info from tRNS
	var idatBuf bytes.Buffer  // concatenated IDAT payload

	for {
		chunk, err := pr.readChunkHeader()
		if err != nil {
			return nil, err
		}

		switch chunk.typ {
		case chunkIHDR:
			if chunk.length != 13 {
				return nil, fmt.Errorf("invalid IHDR length: %d", chunk.length)
			}
			var b [13]byte
			if _, err := io.ReadFull(pr.r, b[:]); err != nil {
				return nil, err
			}
			ihdr = parseIHDR(b[:])
			if err := pr.skipCRC(); err != nil {
				return nil, err
			}
			if ihdr.compression != 0 || ihdr.filterMethod != 0 {
				return nil, errors.New("unsupported compression/filter method")
			}
			if ihdr.interlace != 0 {
				return nil, errors.New("interlaced PNG not supported in MVP")
			}
			if ihdr.bitDepth != 8 {
				return nil, fmt.Errorf("bit depth %d not supported in MVP", ihdr.bitDepth)
			}
			if err := validateColorType(ihdr.colorType); err != nil {
				return nil, err
			}

		case chunkPLTE:
			// length must be multiple of 3, up to 256*3
			if chunk.length%3 != 0 || chunk.length == 0 || chunk.length > 256*3 {
				return nil, fmt.Errorf("invalid PLTE length: %d", chunk.length)
			}
			buf := make([]byte, int(chunk.length))
			if _, err := io.ReadFull(pr.r, buf); err != nil {
				return nil, err
			}
			if err := pr.skipCRC(); err != nil {
				return nil, err
			}
			n := int(chunk.length / 3)
			palette = make([]color.NRGBA, n)
			for i := 0; i < n; i++ {
				palette[i] = color.NRGBA{R: buf[i*3+0], G: buf[i*3+1], B: buf[i*3+2], A: 0xFF}
			}

		case chunktRNS:
			// tRNS format depends on color type; store bytes and interpret later
			buf := make([]byte, int(chunk.length))
			if _, err := io.ReadFull(pr.r, buf); err != nil {
				return nil, err
			}
			if err := pr.skipCRC(); err != nil {
				return nil, err
			}
			trns = parseTRNS(buf)

		case chunkIDAT:
			// accumulate payload
			if _, err := io.CopyN(&idatBuf, pr.r, int64(chunk.length)); err != nil {
				return nil, err
			}
			if err := pr.skipCRC(); err != nil {
				return nil, err
			}

		case chunkIEND:
			// done; decode the image from accumulated IDATs
			if err := pr.skipPayloadAndCRC(0); err != nil {
				return nil, err
			}
			return decodeImageData(ihdr, palette, trns, &idatBuf, cb)

		default:
			// Skip unhandled chunks
			if err := pr.skipPayloadAndCRC(int64(chunk.length)); err != nil {
				return nil, err
			}
		}
	}
}

// DecodeImage is a convenience wrapper that returns image.Image.
func DecodeImage(r io.Reader) (image.Image, error) {
	img, err := Decode(r, nil)
	if err != nil {
		return nil, err
	}
	return img, nil
}

// ---- Internal implementation ----

var pngSig = [8]byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}

const (
	chunkIHDR uint32 = 0x49484452 // "IHDR"
	chunkPLTE uint32 = 0x504C5445 // "PLTE"
	chunkIDAT uint32 = 0x49444154 // "IDAT"
	chunkIEND uint32 = 0x49454E44 // "IEND"
	chunktRNS uint32 = 0x74524E53 // "tRNS"
)

type chunkHeader struct {
	length uint32
	typ    uint32
}

type pngReader struct {
	r io.Reader
}

func (p *pngReader) readSignature() error {
	var s [8]byte
	if _, err := io.ReadFull(p.r, s[:]); err != nil {
		return err
	}
	if s != pngSig {
		return errors.New("invalid PNG signature")
	}
	return nil
}

func (p *pngReader) readChunkHeader() (chunkHeader, error) {
	var hdr chunkHeader
	var b [8]byte
	if _, err := io.ReadFull(p.r, b[:]); err != nil {
		return hdr, err
	}
	hdr.length = binary.BigEndian.Uint32(b[0:4])
	hdr.typ = binary.BigEndian.Uint32(b[4:8])
	return hdr, nil
}

func (p *pngReader) skipCRC() error {
	var crc [4]byte
	_, err := io.ReadFull(p.r, crc[:])
	return err
}

func (p *pngReader) skipPayloadAndCRC(n int64) error {
	if n > 0 {
		if _, err := io.CopyN(io.Discard, p.r, n); err != nil {
			return err
		}
	}
	return p.skipCRC()
}

type ihdrData struct {
	w, h         int
	bitDepth     uint8
	colorType    uint8
	compression  uint8
	filterMethod uint8
	interlace    uint8
}

func parseIHDR(b []byte) ihdrData {
	return ihdrData{
		w:            int(binary.BigEndian.Uint32(b[0:4])),
		h:            int(binary.BigEndian.Uint32(b[4:8])),
		bitDepth:     b[8],
		colorType:    b[9],
		compression:  b[10],
		filterMethod: b[11],
		interlace:    b[12],
	}
}

func validateColorType(ct uint8) error {
	switch ct {
	case 0, 2, 3, 4, 6:
		return nil
	default:
		return fmt.Errorf("unsupported color type: %d", ct)
	}
}

type tRNSData struct {
	// For color type 3 (indexed), alpha for each palette entry
	palAlpha []uint8
	// For color type 0 (gray), a single 16-bit gray value (MVP uses 8-bit)
	grayKey *uint16
	// For color type 2 (truecolor), a single 16-bit R,G,B key (MVP uses 8-bit)
	rgbKey *[3]uint16
}

func parseTRNS(b []byte) *tRNSData {
	// We don't know color type here; store raw and interpret later.
	// PNG allows tRNS with PLTE (indexed) or with grayscale/truecolor.
	if len(b) == 0 {
		return &tRNSData{}
	}
	return &tRNSData{palAlpha: append([]byte(nil), b...)}
}

func decodeImageData(h ihdrData, palette []color.NRGBA, trns *tRNSData, idat *bytes.Buffer, cb LineCallback) (
	*image.RGBA, error,
) {
	if h.w <= 0 || h.h <= 0 {
		return nil, errors.New("invalid IHDR dimensions")
	}

	zr, err := zlib.NewReader(bytes.NewReader(idat.Bytes()))
	if err != nil {
		return nil, fmt.Errorf("zlib: %w", err)
	}
	defer zr.Close()

	spp := samplesPerPixel(h.colorType)
	if spp == 0 {
		return nil, fmt.Errorf("unsupported color type %d", h.colorType)
	}

	rowBytes := h.w * spp // 8-bit only in MVP
	prev := make([]byte, rowBytes)
	cur := make([]byte, rowBytes)

	out := image.NewRGBA(image.Rect(0, 0, h.w, h.h))

	// Prepare transparency helpers
	var alphaForIndex []uint8
	var grayKey8 *uint8
	var rgbKey8 *[3]uint8
	if trns != nil {
		// Interpret based on color type
		switch h.colorType {
		case 3: // indexed
			alphaForIndex = make([]byte, len(palette))
			// Default alpha 255; tRNS supplies N entries overriding
			for i := range alphaForIndex {
				alphaForIndex[i] = 0xFF
			}
			for i := 0; i < len(trns.palAlpha) && i < len(alphaForIndex); i++ {
				alphaForIndex[i] = trns.palAlpha[i]
			}
		case 0: // gray key stored as 2 bytes (16-bit); MVP approximates to 8-bit
			if len(trns.palAlpha) >= 2 {
				v := binary.BigEndian.Uint16(trns.palAlpha[:2])
				vv := uint8(v >> 8) // approximate by high byte
				grayKey8 = &vv
			}
		case 2: // RGB key (6 bytes)
			if len(trns.palAlpha) >= 6 {
				var k [3]uint8
				k[0] = uint8(binary.BigEndian.Uint16(trns.palAlpha[0:2]) >> 8)
				k[1] = uint8(binary.BigEndian.Uint16(trns.palAlpha[2:4]) >> 8)
				k[2] = uint8(binary.BigEndian.Uint16(trns.palAlpha[4:6]) >> 8)
				rgbKey8 = &k
			}
		}
	}

	bpp := bytesPerPixel(h)
	if bpp == 0 {
		return nil, errors.New("internal: bpp=0")
	}

	for y := 0; y < h.h; y++ {
		// Each row starts with a filter type byte
		var f [1]byte
		if _, err := io.ReadFull(zr, f[:]); err != nil {
			return nil, fmt.Errorf("read filter: %w", err)
		}
		if _, err := io.ReadFull(zr, cur); err != nil {
			return nil, fmt.Errorf("read row: %w", err)
		}

		if err := unfilter(cur, prev, bpp, int(f[0])); err != nil {
			return nil, err
		}

		// Convert to RGBA into out.Pix for row y
		rowRGBA := out.Pix[y*out.Stride : y*out.Stride+h.w*4]

		switch h.colorType {
		case 6: // RGBA
			copy(rowRGBA, expandRGBA(cur))
		case 2: // RGB -> RGBA (opaque or RGB key transparent)
			for i, x := 0, 0; x < h.w; x++ {
				r := cur[i+0]
				g := cur[i+1]
				b := cur[i+2]
				a := uint8(0xFF)
				if rgbKey8 != nil && r == rgbKey8[0] && g == rgbKey8[1] && b == rgbKey8[2] {
					a = 0
				}
				j := x * 4
				rowRGBA[j+0] = r
				rowRGBA[j+1] = g
				rowRGBA[j+2] = b
				rowRGBA[j+3] = a
				i += 3
			}
		case 0: // Gray -> RGBA
			for x := 0; x < h.w; x++ {
				g := cur[x]
				a := uint8(0xFF)
				if grayKey8 != nil && g == *grayKey8 {
					a = 0
				}
				j := x * 4
				rowRGBA[j+0] = g
				rowRGBA[j+1] = g
				rowRGBA[j+2] = g
				rowRGBA[j+3] = a
			}
		case 3: // Indexed -> RGBA
			if len(palette) == 0 {
				return nil, errors.New("indexed PNG missing PLTE")
			}
			for x := 0; x < h.w; x++ {
				idx := int(cur[x])
				if idx >= len(palette) {
					return nil, fmt.Errorf("palette index %d out of range", idx)
				}
				c := palette[idx]
				a := c.A
				if alphaForIndex != nil && idx < len(alphaForIndex) {
					a = alphaForIndex[idx]
				}
				j := x * 4
				rowRGBA[j+0] = c.R
				rowRGBA[j+1] = c.G
				rowRGBA[j+2] = c.B
				rowRGBA[j+3] = a
			}
		case 4: // Gray+Alpha (GA)
			for i, x := 0, 0; x < h.w; x++ {
				g := cur[i]
				a := cur[i+1]
				j := x * 4
				rowRGBA[j+0] = g
				rowRGBA[j+1] = g
				rowRGBA[j+2] = g
				rowRGBA[j+3] = a
				i += 2
			}
		default:
			return nil, fmt.Errorf("unsupported color type in MVP: %d", h.colorType)
		}

		if cb != nil {
			cb(y, rowRGBA)
		}
		// swap buffers for next line
		prev, cur = cur, prev
	}

	return out, nil
}

func samplesPerPixel(ct uint8) int {
	switch ct {
	case 0:
		return 1 // Gray
	case 2:
		return 3 // RGB
	case 3:
		return 1 // Indexed
	case 4:
		return 2 // Gray+Alpha
	case 6:
		return 4 // RGBA
	default:
		return 0
	}
}

func bytesPerPixel(h ihdrData) int {
	// 8-bit only MVP
	return samplesPerPixel(h.colorType)
}

func unfilter(cur, prev []byte, bpp int, f int) error {
	switch f {
	case 0: // None
		return nil
	case 1: // Sub
		for i := 0; i < len(cur); i++ {
			left := byte(0)
			if i >= bpp {
				left = cur[i-bpp]
			}
			cur[i] = cur[i] + left
		}
		return nil
	case 2: // Up
		for i := 0; i < len(cur); i++ {
			cur[i] = cur[i] + prev[i]
		}
		return nil
	case 3: // Average
		for i := 0; i < len(cur); i++ {
			left := byte(0)
			if i >= bpp {
				left = cur[i-bpp]
			}
			top := prev[i]
			cur[i] = cur[i] + byte(((int(left)+int(top))>>1)&0xFF)
		}
		return nil
	case 4: // Paeth
		for i := 0; i < len(cur); i++ {
			var a, b, c byte // left, up, up-left
			if i >= bpp {
				a = cur[i-bpp]
			}
			b = prev[i]
			if i >= bpp {
				c = prev[i-bpp]
			}
			cur[i] = cur[i] + paeth(a, b, c)
		}
		return nil
	default:
		return fmt.Errorf("unknown filter type %d", f)
	}
}

func paeth(a, b, c byte) byte {
	pa := int(a)
	pb := int(b)
	pc := int(c)
	p := pa + pb - pc
	pa_ := abs(p - pa)
	pb_ := abs(p - pb)
	pc_ := abs(p - pc)
	if pa_ <= pb_ && pa_ <= pc_ {
		return a
	}
	if pb_ <= pc_ {
		return b
	}
	return c
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func expandRGBA(src []byte) []byte {
	// src is already RGBA per-pixel, copy per row is fine.
	return src
}
