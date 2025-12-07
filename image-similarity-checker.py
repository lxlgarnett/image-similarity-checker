import argparse
import os
from PIL import Image
import numpy as np
import scipy.fftpack
from itertools import combinations
from collections import defaultdict


def find_images(directory: str, extensions=None):
    """Finds all image files in a directory recursively.

    Args:
        directory: The path to the directory to search.
        extensions: Iterable of lowercase file extensions to include (e.g.,
            [".jpg", ".png"]). Defaults to common image formats.

    Yields:
        The full path to each image file found.
    """
    image_extensions = extensions or [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                yield os.path.join(root, file)


def phash_hex(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Computes the perceptual hash (pHash) of an image.

    The hash is based on the Discrete Cosine Transform (DCT) of the image.
    It is a robust fingerprint that can be used to compare images for similarity.

    Args:
        image: The PIL.Image object to hash.
        hash_size: The size of the hash to generate. A larger size means a more
            detailed hash, but it may be more sensitive to small changes.
        highfreq_factor: The factor by which to scale the image size before
            applying the DCT. A higher factor includes more high-frequency
            components.

    Returns:
        A hexadecimal string representing the perceptual hash of the image.

    Raises:
        ValueError: If hash_size is less than 2.
    """
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
    """Calculates the Hamming distance between two strings.

    Args:
        s1: The first string.
        s2: The second string.

    Returns:
        The number of positions at which the characters are different.
    """
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def main():
    """Finds and groups similar images in a directory."""
    parser = argparse.ArgumentParser(
        description="Finds and groups similar images in a directory based on perceptual hash."
    )
    parser.add_argument("directory", help="The directory to search for images.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="The Hamming distance threshold for similarity. Default is 5.",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=8,
        help=(
            "The size of the perceptual hash. Larger values capture more detail but are "
            "more sensitive to changes. Must be at least 2. Default is 8."
        ),
    )
    parser.add_argument(
        "--highfreq-factor",
        type=int,
        default=4,
        help=(
            "Factor to scale the image before DCT is applied. Higher values include more "
            "high-frequency components. Must be at least 1. Default is 4."
        ),
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        help=(
            "Space-separated list of file extensions to include (e.g., .jpg .png). "
            "Defaults to common image formats."
        ),
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found at '{args.directory}'")
        return

    if args.threshold < 0:
        print("Error: Threshold must be a non-negative integer.")
        return

    if args.hash_size < 2:
        print("Error: hash_size must be at least 2.")
        return

    if args.highfreq_factor < 1:
        print("Error: highfreq_factor must be at least 1.")
        return

    if args.extensions:
        normalized_exts = []
        for ext in args.extensions:
            if not ext.startswith("."):
                print("Error: Extensions must start with a dot (e.g., .jpg)")
                return
            normalized_exts.append(ext.lower())
        extensions = normalized_exts
    else:
        extensions = None

    print("Finding images...")
    image_paths = list(find_images(args.directory, extensions=extensions))
    if not image_paths:
        print("No images found.")
        return

    print(f"Found {len(image_paths)} images. Calculating hashes...")
    hashes = {}
    for path in image_paths:
        try:
            image = Image.open(path)
            hashes[path] = phash_hex(
                image, hash_size=args.hash_size, highfreq_factor=args.highfreq_factor
            )
        except Exception as e:
            print(f"Could not process {path}: {e}")

    print("Comparing images...")
    # Parent pointer for Union-Find
    parent = {path: path for path in hashes}
    # Pre-calculate binary hashes to avoid repeated conversions during comparisons
    binary_hashes = {path: hex_to_binary(h) for path, h in hashes.items()}

    def find_set(path):
        if parent[path] == path:
            return path
        parent[path] = find_set(parent[path])
        return parent[path]

    def unite_sets(path1, path2):
        root1 = find_set(path1)
        root2 = find_set(path2)
        if root1 != root2:
            parent[root2] = root1

    # Compare all pairs of images
    for path1, path2 in combinations(hashes.keys(), 2):
        hash1 = binary_hashes[path1]
        hash2 = binary_hashes[path2]
        if hamming_distance(hash1, hash2) <= args.threshold:
            unite_sets(path1, path2)

    # Group images by their root parent
    groups = defaultdict(list)
    for path in hashes:
        root = find_set(path)
        groups[root].append(path)

    print("\n--- Similarity Report ---")
    similar_groups_found = False
    group_count = 0
    for root in groups:
        group_paths = groups[root]
        if len(group_paths) > 1:
            similar_groups_found = True
            group_count += 1
            # The root image of the group
            root_hash_binary = binary_hashes[root]

            print(f"\nGroup {group_count} (Root: {root}):")

            # Sort paths for consistent output
            for path in sorted(group_paths):
                path_hash_binary = binary_hashes[path]
                distance = hamming_distance(root_hash_binary, path_hash_binary)
                hash_length = len(root_hash_binary)
                similarity = 100 - (distance / hash_length * 100)
                print(f"- {similarity:.2f}% similar: {path}")

    if not similar_groups_found:
        print("No similar images found with the current threshold.")


if __name__ == "__main__":
    main()
