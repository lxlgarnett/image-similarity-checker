"""
Finds and groups similar images in a directory based on perceptual hash.
"""
import argparse
from collections import defaultdict
from itertools import combinations
import os

import numpy as np
from PIL import Image
import scipy.fftpack

from models.data_classes import ReportData
from models.constants import DEFAULT_IMAGE_EXTENSIONS


def find_images(directory: str, extensions=None, exclude_paths=None):
    """Finds all image files in a directory recursively.

    Args:
        directory: The path to the directory to search.
        extensions: Iterable of lowercase file extensions to include (e.g.,
            [".jpg", ".png"]). Defaults to common image formats.
        exclude_paths: Iterable of absolute directory paths to skip while walking
            the tree.

    Yields:
        The full path to each image file found.
    """
    image_extensions = extensions or DEFAULT_IMAGE_EXTENSIONS
    excluded = set(exclude_paths or [])

    for root, dirs, files in os.walk(directory):
        dirs[:] = [
            d
            for d in dirs
            if not any(
                os.path.commonpath([os.path.join(root, d), exclude_dir])
                == exclude_dir
                for exclude_dir in excluded
            )
        ]
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


def _get_argument_parser():
    """Initializes and returns the argument parser for the script."""
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
            "Defaults to common image formats: "
            + ", ".join(DEFAULT_IMAGE_EXTENSIONS)
        ),
    )
    parser.add_argument(
        "--show-unique",
        action="store_true",
        help="Also list images that did not meet the similarity threshold.",
    )
    parser.add_argument(
        "--output",
        help="Optional file path to save the similarity report alongside console output.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        metavar="PATH",
        help=(
            "One or more directories to skip while searching for images. "
            "Paths may be absolute or relative to the target directory."
        ),
    )
    return parser


def _validate_and_prepare_args(args):
    """Validates arguments and prepares paths.

    Args:
        args: The parsed arguments from argparse.

    Returns:
        A tuple of (target_directory, extensions, exclude_paths, output_path).
        Returns None if validation fails.
    """
    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found at '{args.directory}'")
        return None
    target_directory = os.path.abspath(args.directory)

    validation_rules = {
        (args.threshold < 0, "Threshold must be a non-negative integer."),
        (args.hash_size < 2, "hash_size must be at least 2."),
        (args.highfreq_factor < 1, "highfreq_factor must be at least 1."),
    }
    for rule, message in validation_rules:
        if rule:
            print(f"Error: {message}")
            return None

    extensions = None
    if args.extensions:
        if any(not ext.startswith(".") for ext in args.extensions):
            print("Error: Extensions must start with a dot (e.g., .jpg)")
            return None
        extensions = [ext.lower() for ext in args.extensions]

    exclude_paths = []
    if args.exclude:
        for path in args.exclude:
            abs_path = (
                path
                if os.path.isabs(path)
                else os.path.abspath(os.path.join(target_directory, path))
            )
            if not os.path.isdir(abs_path):
                print(f"Error: Exclude directory not found at '{path}'")
                return None
            exclude_paths.append(abs_path)

    output_path = None
    if args.output:
        output_path = os.path.abspath(args.output)
        if os.path.isdir(output_path):
            print("Error: Output path refers to a directory; please provide a file path.")
            return None
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                print(f"Error creating output directory '{output_dir}': {e}")
                return None

    return target_directory, extensions, exclude_paths, output_path


def _calculate_hashes(image_paths, hash_size, highfreq_factor, log):
    """Calculates the perceptual hashes for a list of images.

    Args:
        image_paths: A list of paths to image files.
        hash_size: The size of the hash to generate.
        highfreq_factor: The factor for high-frequency components.
        log: A function to log messages.

    Returns:
        A dictionary mapping image paths to their perceptual hashes.
    """
    log(f"Found {len(image_paths)} images. Calculating hashes...")
    hashes = {}
    for path in image_paths:
        try:
            with Image.open(path) as image:
                hashes[path] = phash_hex(
                    image,
                    hash_size=hash_size,
                    highfreq_factor=highfreq_factor,
                )
        except (IOError, ValueError) as e:
            log(f"Could not process {path}: {e}")
    return hashes


def _group_images_by_similarity(hashes, threshold, log):
    """Groups images based on hash similarity using a Union-Find data structure.

    Args:
        hashes: A dictionary mapping image paths to their hex hashes.
        threshold: The Hamming distance threshold for similarity.
        log: A function to log messages.

    Returns:
        A tuple containing:
        - A dictionary grouping similar image paths.
        - A dictionary mapping image paths to their binary hashes.
    """
    log("Comparing images...")
    parent = {path: path for path in hashes}
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

    for path1, path2 in combinations(hashes.keys(), 2):
        hash1 = binary_hashes[path1]
        hash2 = binary_hashes[path2]
        if hamming_distance(hash1, hash2) <= threshold:
            unite_sets(path1, path2)

    groups = defaultdict(list)
    for path in hashes:
        root = find_set(path)
        groups[root].append(path)

    return groups, binary_hashes


def _generate_and_save_report(report_data):
    """Generates and saves the similarity report.

    Args:
        report_data: A ReportData object containing report information.
    """
    report_data.log("\n--- Similarity Report ---")
    similar_groups_found = False
    group_count = 0
    for root, group_paths in report_data.groups.items():
        if len(group_paths) > 1:
            similar_groups_found = True
            group_count += 1
            root_hash_binary = report_data.binary_hashes[root]

            report_data.log(f"\nGroup {group_count} (Root: {root}):")

            for path in sorted(group_paths):
                path_hash_binary = report_data.binary_hashes[path]
                distance = hamming_distance(root_hash_binary, path_hash_binary)
                hash_length = len(root_hash_binary)
                similarity = 100 - (distance / hash_length * 100)
                report_data.log(f"- {similarity:.2f}% similar: {path}")

    if not similar_groups_found:
        report_data.log("No similar images found with the current threshold.")

    unique_images = [
        path
        for paths in report_data.groups.values()
        for path in paths
        if len(paths) == 1
    ]

    if report_data.show_unique:
        report_data.log("\n--- Unique Images ---")
        if unique_images:
            for path in sorted(unique_images):
                report_data.log(f"- {path}")
        else:
            report_data.log("No unique images found.")

    report_data.log("\n--- Summary ---")
    report_data.log(f"Total images processed: {report_data.all_images_count}")
    report_data.log(f"Similar groups found: {group_count}")
    report_data.log(f"Unique images: {len(unique_images)}")

    if report_data.output_path:
        try:
            with open(report_data.output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_data.report_lines))
            report_data.log(f"\nReport saved to {report_data.output_path}")
        except OSError as e:
            report_data.log(f"\nError writing report to {report_data.output_path}: {e}")


def main():
    """Finds and groups similar images in a directory."""
    parser = _get_argument_parser()
    args = parser.parse_args()

    validated_args = _validate_and_prepare_args(args)
    if not validated_args:
        return

    target_directory, extensions, exclude_paths, output_path = validated_args

    report_lines = []

    def log(line: str = ""):
        print(line)
        report_lines.append(line)

    log("Finding images...")
    image_paths = list(
        find_images(
            target_directory,
            extensions=extensions,
            exclude_paths=exclude_paths,
        )
    )
    if not image_paths:
        log("No images found.")
        return

    hashes = _calculate_hashes(
        image_paths, args.hash_size, args.highfreq_factor, log
    )

    groups, binary_hashes = _group_images_by_similarity(
        hashes, args.threshold, log
    )

    report_data = ReportData(
        groups=groups,
        binary_hashes=binary_hashes,
        all_images_count=len(hashes),
        show_unique=args.show_unique,
        output_path=output_path,
        report_lines=report_lines,
        log=log,
    )
    _generate_and_save_report(report_data)


if __name__ == "__main__":
    main()
