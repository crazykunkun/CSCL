from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def _gray(image: Image.Image) -> Image.Image:
    return ImageOps.grayscale(image.convert("RGB"))


def pseudo_srm(image: Image.Image) -> Image.Image:
    gray = _gray(image)
    blur = gray.filter(ImageFilter.GaussianBlur(radius=1.2))
    arr = np.asarray(gray, dtype=np.float32) - np.asarray(blur, dtype=np.float32)
    arr = np.abs(arr)
    arr = arr / (arr.max() + 1e-8) * 255
    return Image.fromarray(arr.astype(np.uint8))


def pseudo_dct(image: Image.Image) -> Image.Image:
    gray = _gray(image).resize((256, 256))
    arr = np.asarray(gray, dtype=np.float32)
    gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
    gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :]))
    energy = gx + gy
    energy = energy / (energy.max() + 1e-8) * 255
    return Image.fromarray(energy.astype(np.uint8)).resize(image.size)


def pseudo_fft(image: Image.Image) -> Image.Image:
    gray = _gray(image).resize((256, 256))
    arr = np.asarray(gray, dtype=np.float32)
    freq = np.fft.fftshift(np.fft.fft2(arr - arr.mean()))
    h, w = arr.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    mask = radius > min(h, w) * 0.18
    high = np.fft.ifft2(np.fft.ifftshift(freq * mask)).real
    high = np.abs(high)
    high = high / (high.max() + 1e-8) * 255
    return Image.fromarray(high.astype(np.uint8)).resize(image.size)


def pseudo_haar(image: Image.Image) -> Image.Image:
    gray = _gray(image).resize((256, 256))
    arr = np.asarray(gray, dtype=np.float32)
    even_rows = arr[0::2, :]
    odd_rows = arr[1::2, :]
    hpass_v = np.abs(even_rows[: odd_rows.shape[0], :] - odd_rows)
    even_cols = arr[:, 0::2]
    odd_cols = arr[:, 1::2]
    hpass_h = np.abs(even_cols[:, : odd_cols.shape[1]] - odd_cols)
    hpass_v = np.repeat(hpass_v, 2, axis=0)[:256, :]
    hpass_h = np.repeat(hpass_h, 2, axis=1)[:, :256]
    out = hpass_v + hpass_h
    out = out / (out.max() + 1e-8) * 255
    return Image.fromarray(out.astype(np.uint8)).resize(image.size)


def make_frequency_views(image: Image.Image) -> dict[str, Image.Image]:
    return {
        "SRM 残差近似图": pseudo_srm(image),
        "DCT 高频能量近似图": pseudo_dct(image),
        "FFT 高通响应近似图": pseudo_fft(image),
        "Haar 小波高频近似图": pseudo_haar(image),
    }
