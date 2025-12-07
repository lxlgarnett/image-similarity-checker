"""
Data classes for the image similarity checker.
"""
from collections import namedtuple

ReportData = namedtuple(
    "ReportData",
    [
        "groups",
        "binary_hashes",
        "all_images_count",
        "show_unique",
        "output_path",
        "report_lines",
        "log",
    ],
)
