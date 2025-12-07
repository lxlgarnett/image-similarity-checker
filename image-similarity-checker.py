import argparse
import os
from PIL import Image
import numpy as np
import scipy.fftpack
from itertools import combinations
from collections import defaultdict


def find_images(directory: str):
    """Finds all image files in a directory recursively."""
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                yield os.path.join(root, file)


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
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found at '{args.directory}'")
        return

    print("Finding images...")
    image_paths = list(find_images(args.directory))
    if not image_paths:
        print("No images found.")
        return

    print(f"Found {len(image_paths)} images. Calculating hashes...")
    hashes = {}
    for path in image_paths:
        try:
            image = Image.open(path)
            hashes[path] = phash_hex(image)
        except Exception as e:
            print(f"Could not process {path}: {e}")

    print("Comparing images...")
    # Parent pointer for Union-Find
    parent = {path: path for path in hashes}

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
        hash1 = hex_to_binary(hashes[path1])
        hash2 = hex_to_binary(hashes[path2])
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
    # Pre-calculate binary hashes to avoid re-computation
    binary_hashes = {path: hex_to_binary(h) for path, h in hashes.items()}

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
