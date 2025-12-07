# Image Similarity Checker

A Python script to find and group similar images within a directory using a perceptual hashing algorithm (pHash). This tool is useful for identifying duplicate or near-duplicate images in a large collection.

## Features

- **Recursive Search:** Scans a directory and all its subdirectories for images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`, `.tiff`, `.tif`).
- **Perceptual Hashing:** Generates a robust "fingerprint" for each image that represents its visual content.
- **Similarity Detection:** Compares all images against each other using Hamming distance to find images that are visually similar.
- **Configurable Threshold:** Allows you to set the sensitivity of the similarity detection with a `--threshold` flag.
- **Smart Grouping:** All images that are similar to each other (e.g., A is similar to B, and B is similar to C) are collected into a single group.
- **Clear Reporting:** Provides a clean report of the discovered groups, including the similarity percentage of each image relative to a root image in its group.

## Requirements

- Python 3.6+
- Dependencies listed in `requirements.txt` (`Pillow`, `numpy`, `scipy`)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/image-similarity-checker.git
    cd image-similarity-checker
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script by providing the path to the directory you want to scan.

```bash
python image-similarity-checker.py /path/to/your/image_directory
```

### Options

You can adjust the similarity sensitivity using the `--threshold` flag. This value represents the maximum **Hamming distance** allowed between two perceptual hashes for them to be considered "similar."

-   **Hamming Distance:** In the context of pHash, Hamming distance quantifies the number of bits that differ between two image hashes. A smaller Hamming distance indicates greater similarity.
-   **Impact of Threshold:**
    *   A **lower threshold** (e.g., `0` to `5`) means images must be very close visually to be grouped. `0` will only group identical images.
    *   A **higher threshold** (e.g., `10` or more) will group images with more noticeable visual differences.
    *   Small variations like minor resizing, cropping, or compression often result in Hamming distances between `1` and `5`.
-   **Choosing a Threshold:** The optimal threshold depends on your specific needs and the types of image variations you want to detect. Experimentation is recommended. A common starting point for finding near-duplicates is a threshold of `5`.

**Default Threshold:** The default value for `--threshold` is `5`.

```bash
python image-similarity-checker.py /path/to/your/images --threshold 3
```

Additional configuration flags allow you to control how the perceptual hash is generated:

-   `--hash-size`: Size of the perceptual hash. Larger values capture more detail but are more sensitive to visual changes. Must be at least 2. (Default: `8`)
-   `--highfreq-factor`: Factor used to scale the image before applying the DCT. Higher values include more high-frequency components. Must be at least 1. (Default: `4`)
-   `--extensions`: Space-separated list of file extensions to include (e.g., `.jpg .png`). Defaults to common image formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`, `.tiff`, `.tif`).
-   `--show-unique`: Include a section listing images that did not meet the similarity threshold.
-   `--output`: Optional path to save the similarity report to a file while also printing to the console. Parent directories are
    created automatically when needed, and the path must refer to a file (not a directory).
-   `--exclude`: One or more directories to skip when searching for images. Paths can be absolute or relative to the target directory.
-   The report always concludes with a summary showing how many images were processed, how many similar groups were found, and how many images were unique.

Example with custom hash parameters:

```bash
python image-similarity-checker.py /path/to/your/images --hash-size 12 --highfreq-factor 5 --extensions .jpg .png
```

To also list images that did not meet the similarity threshold, include `--show-unique`:

```bash
python image-similarity-checker.py /path/to/your/images --show-unique
```

## Example Output

```
$ python image-similarity-checker.py ./my_images
Finding images...
Found 150 images. Calculating hashes...
Comparing images...

--- Similarity Report ---

Group 1 (Root: ./my_images/cats/fluffy_cat.jpg):
- 100.00% similar: ./my_images/cats/fluffy_cat.jpg
- 98.44% similar: ./my_images/archive/cat_pic_2022.jpg
- 96.88% similar: ./my_images/cats/fluffy_cat_small.png

Group 2 (Root: ./my_images/landscapes/mountain_view.png):
- 100.00% similar: ./my_images/landscapes/mountain_view.png
- 99.22% similar: ./my_images/vacation/IMG_4032.jpeg

No other similar images found with the current threshold.
```