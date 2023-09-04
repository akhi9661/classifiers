# Land Cover Classification with Random Forest

This project provides a set of Python functions for land cover classification using Random Forest. It includes utilities for generating training points, training a Random Forest classifier, performing classification, and evaluating the results.

## Getting Started

Follow the steps below to get started with this project.

### Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3.x
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Rasterio](https://rasterio.readthedocs.io/en/latest/)
- [scikit-learn](https://scikit-learn.org/)
- [Fiona](https://fiona.readthedocs.io/en/stable/)
- [Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install)

You can install these dependencies using pip:

---
### Usage

To classify land cover using Random Forest, follow these steps:

1. Prepare your raster images in the desired format and specify their file paths in the `raster_paths` list.

2. Use the `classify` function to perform land cover classification:

   ```python

   raster_paths = ["image1.tif", "image2.tif"]
   training_sites, test_df, predictions, metrics = classify(
       raster_paths,
       training_points=200,
       uniform_to_random_ratio=0.6,
       train_to_test_ratio=0.8,
       window_size=1,
       z_score_threshold=3,
       force=False
   )

   print(f"Overall Accuracy: {metrics['Overall Accuracy']:.3f}")

---
### Functions
Here are the main functions provided by this project:

- generate_training_sites: Generate training sites with land cover information.
- validate_results: Evaluate classification results and compute metrics.
- test_model: Test the classification model with a set of test points.
- raster_to_df: Convert raster images to DataFrames.
- rf_classifier: Train a Random Forest classifier and perform classification.
- subset_test_points: Generate a subset of test points from training points.

---
### Example
Check the example in the Usage section for a sample implementation.

---
### License
This project is licensed under the MIT License - see the LICENSE file for details.

---
### Acknowledgments
- NumPy for numerical operations.
- Pandas for data manipulation.
- Rasterio for raster data handling.
- scikit-learn for machine learning tools.
- Fiona for vector data handling.
- Google Earth Engine Python API for Earth Engine integration.

Feel free to contribute, report issues, or suggest improvements to this project!
