# -*- coding: utf-8 -*-
import argparse
from PIL import Image
import sys
import numpy as np
import scipy.fftpack


def phash_hex(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    if hash_size < 2:
        raise ValueError("hash_size must be >= 2")

    img_size = hash_size * highfreq_factor
    image = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)

    pixels = np.asarray(image, dtype=np.float32)
    dct_cols = scipy.fftpack.dct(pixels, axis=0)
    dct = scipy.fftpack.dct(dct_cols, axis=1)

    dct_lowfreq = dct[:hash_size, :hash_size]
    med = np.median(dct_lowfreq)
    bits = (dct_lowfreq > med).flatten()

    hex_str = ""
    for i in range(0, len(bits), 4):
        nibble = bits[i : i + 4]
        b = "".join("1" if x else "0" for x in nibble)
        hex_str += f"{int(b, 2):x}"

    return hex_str


def hex_to_binary(hex_string: str) -> str:
    """Converts a hex string to a binary string."""
    return bin(int(hex_string, 16))[2:].zfill(len(hex_string) * 4)


def hamming_distance(s1: str, s2: str) -> int:
    """Calculates the Hamming distance between two strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def main():
    """Compares two images for similarity using perceptual hash."""
    parser = argparse.ArgumentParser(
        description="Compare two images for similarity using perceptual hash."
    )
    parser.add_argument("image1", help="The first image file.")
    parser.add_argument("image2", help="The second image file.")
    args = parser.parse_args()

    hash1_hex = phash_hex(Image.open(args.image1))
    hash2_hex = phash_hex(Image.open(args.image2))

    hash1_binary = hex_to_binary(hash1_hex)
    hash2_binary = hex_to_binary(hash2_hex)

    distance = hamming_distance(hash1_binary, hash2_binary)

    print(f"Hash 1 (hex): {hash1_hex}")
    print(f"Hash 2 (hex): {hash2_hex}")
    print(f"Hamming distance: {distance}")
    print(
        f"Similarity: {100 - (distance / len(hash1_binary) * 100):.2f}%"
    )


if __name__ == "__main__":
    main()
