import ee, random, os, rasterio
from osgeo import gdal, osr
import pandas as pd, numpy as np
from pyproj import Transformer
from shapely.geometry import Point
from geopandas import GeoDataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from scipy import stats

if not ee.data._credentials:
    ee.Authenticate()
if not ee.data._initialized:
    ee.Initialize()
    
    
def convert_raster_crs_to_wgs84(rast_path):
    """
    Reprojects a raster to the Geographic Coordinate System (GCS) WGS 84 (EPSG:4326) and returns the extent.

    Parameters:
        rast_path (str): The path to the input raster file.

    Returns:
        list: A list containing the extent coordinates in WGS 84 (min_x, min_y, max_x, max_y).
              Returns None if the input raster cannot be opened or if the transformation fails.

    Example:
        extent = convert_raster_crs_to_wgs84('input_raster.tif')
        if extent is not None:
            print(f'Extent in WGS 84: {extent}')
        else:
            print('Failed to reproject the raster.')

    Note:
        This function uses the GDAL library for raster processing.

    References:
        - GDAL: https://gdal.org/
        - EPSG:4326 (WGS 84): https://epsg.io/4326
    """

    print('Processing: Reprojecting to GCS WGS 84 ...')
    raster_dataset = gdal.Open(rast_path)
    if raster_dataset is None:
        return None

    # Get the existing CRS
    rast_crs = raster_dataset.GetProjection()

    # Create a transformation object to convert to GCS WGS 84
    target_crs = osr.SpatialReference()
    target_crs.ImportFromEPSG(4326)  # EPSG code for WGS 84
    transformation = osr.CoordinateTransformation(osr.SpatialReference(wkt=rast_crs), target_crs)

    # Get the extent of the raster
    width = raster_dataset.RasterXSize
    height = raster_dataset.RasterYSize
    gt = raster_dataset.GetGeoTransform()
    min_x = gt[0]
    max_y = gt[3]
    max_x = min_x + gt[1] * width
    min_y = max_y + gt[5] * height

    # Transform the corner coordinates to GCS WGS 84
    transformed_min_x, transformed_min_y, _ = transformation.TransformPoint(min_x, min_y)
    transformed_max_x, transformed_max_y, _ = transformation.TransformPoint(max_x, max_y)

    return [transformed_min_x, transformed_min_y, transformed_max_x, transformed_max_y]

def get_land_cover_class(sample_point, window_size):
    """
    Retrieves the most frequent land cover class within a specified window around a given sample point.

    Parameters:
        sample_point (tuple): A tuple containing the longitude and latitude coordinates of the sample point.
        window_size (float): The half-side length of the square buffer window in degrees (e.g., 0.01 degrees).

    Returns:
        tuple: A tuple containing the longitude, latitude, and the most frequent land cover class as an integer.

    Example:
        sample_point = (78.582894, 29.102852)
        window_size = 0.01  # A window of 0.01 degrees (approximately 1.1 kilometers)
        result = get_land_cover_class(sample_point, window_size)
        print(f'Sample Point: {sample_point}')
        print(f'Most Frequent Land Cover Class: {result[2]}')

    Note:
        This function uses Google Earth Engine (GEE) to access land cover data from the "ESA/WorldCover/v100" ImageCollection.
        The specified window defines the area around the sample point to consider for land cover analysis.

    References:
        - Google Earth Engine (GEE): https://earthengine.google.com/
    """

    longitude, latitude = sample_point
    point = ee.Geometry.Point([longitude, latitude])
    land_cover_collection = ee.ImageCollection("ESA/WorldCover/v100")
    
    def get_most_frequent_value(image):
        # Define a square buffer around the point with a side length of 2*n+1
        buffer = point.buffer(window_size).bounds()
        land_cover = image.reduceRegion(reducer=ee.Reducer.mode(), geometry=buffer)
        return image.set('Map', land_cover.get('Map'))
    
    land_cover_with_class = land_cover_collection.map(get_most_frequent_value)
    most_frequent_land_cover = land_cover_with_class.reduceColumns(ee.Reducer.mode(), ["Map"])
    land_cover_class_int = int(most_frequent_land_cover.get("mode").getInfo())
    
    return longitude, latitude, land_cover_class_int

def generate_grid_points(min_lat, min_lon, max_lat, max_lon, num_points):
    """
    Generate equally spaced grid points within a specified bounding box.

    Parameters:
        min_lat (float): The minimum latitude of the bounding box.
        min_lon (float): The minimum longitude of the bounding box.
        max_lat (float): The maximum latitude of the bounding box.
        max_lon (float): The maximum longitude of the bounding box.
        num_points (int): The desired number of grid points.

    Returns:
        list: A list of tuples representing latitude and longitude coordinates.

    Example:
        points = generate_grid_points(25.0, 75.0, 27.0, 77.0, 16)
        print(points)
    """

    latitude_values = np.linspace(min_lat, max_lat, int(np.sqrt(num_points)))
    longitude_values = np.linspace(min_lon, max_lon, int(np.sqrt(num_points)))

    grid_points = [(lon, lat) for lat in latitude_values for lon in longitude_values]

    if len(grid_points) > num_points:
        grid_points = grid_points[:num_points]

    output = f'Generating {len(grid_points)} equally-spaced points within ' \
             f'[min_lat: {min_lat:.2f}, min_lon: {min_lon:.2f}, ' \
             f'max_lat: {max_lat:.2f}, max_lon: {max_lon:.2f}] ...'
    
    return grid_points


def generate_random_points(min_lat=25, min_lon=75, max_lat=27, max_lon=77, num_points=100):
    """
    Generate random points within a specified bounding box.

    Parameters:
        min_lat (float): The minimum latitude of the bounding box.
        min_lon (float): The minimum longitude of the bounding box.
        max_lat (float): The maximum latitude of the bounding box.
        max_lon (float): The maximum longitude of the bounding box.
        num_points (int): The desired number of random points.

    Returns:
        list: A list of tuples representing latitude and longitude coordinates.

    Example:
        points = generate_random_points(25.0, 75.0, 27.0, 77.0, 100)
        print(points)
    """
    random_points = []
    for _ in range(num_points):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        random_points.append((lon, lat))

    output = f'Generating {len(random_points)} random points within ' \
             f'[min_lat: {min_lat:.2f}, min_lon: {min_lon:.2f}, ' \
             f'max_lat: {max_lat:.2f}, max_lon: {max_lon:.2f}] ...'

    return random_points

def generate_training_sites(rast_paths, training_points=100, uniform_to_random_ratio=0.5, 
                            window_size=1, z_score_threshold=3, force=False, opf=None):
    """
    Generate training sites for land cover classification based on raster data.

    This function generates a set of training sites for land cover classification based on raster data.
    Training sites are sampled at specified locations within the raster extent and include information
    about land cover class, geographic coordinates, and raster values from multiple raster datasets.

    Parameters:
        rast_paths (list): A list of file paths to raster datasets for extracting values.
        training_points (int): The total number of training points to generate.
        uniform_to_random_ratio (float): The ratio of training points to be uniformly distributed
            compared to randomly distributed points. Should be in the range [0, 1].
        window_size (int): The size of the window (in pixels) used for extracting raster values around
            each training point.
        z_score_threshold (int): The Z-score threshold for identifying and removing outliers based on
            raster values. Values above this threshold are considered outliers.
        force (bool): If True, regenerate training sites even if a CSV file with the same parameters exists.
        opf (str): The output folder where the training sites CSV file will be saved.

    Returns:
        pandas.DataFrame: A DataFrame containing training sites with columns for latitude, longitude,
        land cover class, land cover type (LCT), and extracted values from raster datasets.

    Example:
        training_sites = generate_training_sites(rast_paths=['path_to_raster1.tif', 'path_to_raster2.tif'],
                                                 training_points=100, uniform_to_random_ratio=0.5,
                                                 window_size=3, z_score_threshold=2, force=False,
                                                 opf='output_folder/')
        print(training_sites.head())
    """

    print('Processing: Initiating ...')
    raster_extent = convert_raster_crs_to_wgs84(rast_paths[0])
    min_lat, min_lon, max_lat, max_lon = raster_extent
    uniform_num_points = int(training_points * uniform_to_random_ratio)
    random_num_points = int(training_points - uniform_num_points)

    random_points = generate_random_points(min_lat, min_lon, max_lat, max_lon, random_num_points)
    uniform_points = generate_grid_points(min_lat, min_lon, max_lat, max_lon, uniform_num_points)
    total_points = random_points + uniform_points
    
    output_csv = os.path.join(opf, f'training_sites_T-{len(total_points)}_U-{len(uniform_points)}_R-{len(random_points)}.csv')
    if os.path.exists(output_csv) and force is False:
        training_sites = pd.read_csv(output_csv)
        return training_sites
    else: 
        mapping_df = pd.DataFrame({'Value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
                                   'Description': ['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 
                                                   'Built-up', 'Bare / sparse vegetation', 'Snow and ice', 
                                                   'Permanent water bodies', 'Herbaceous wetland', 'Mangroves', 
                                                   'Moss and lichen']})

        training_sites_lulc = []
        print('Processing: Fetching LULC Type from "ESA/WorldCover/v100" ...')
        n = 1

        for sample_point in total_points:
            print(f'Progress: [{n}/{len(total_points)}] processed... ', end = '\r')
            longitude, latitude, land_cover_class = get_land_cover_class(sample_point, window_size)
            training_sites_lulc.append({'longitude': longitude, 'latitude': latitude, 
                                        'land_cover_class': land_cover_class})
            n += 1

        training_sites_v1 = pd.DataFrame(training_sites_lulc)
        training_sites_gee = training_sites_v1.merge(mapping_df, left_on = 'land_cover_class', right_on = 'Value', how = 'left')
        training_sites_gee.drop(['Value'], axis=1, inplace=True)
        training_sites_gee.rename(columns = {'Description': 'LCT'}, inplace=True)

        raster_values_dict = {os.path.basename(raster_path).split('.')[0]: [] for raster_path in raster_paths}
        for raster_path in rast_paths:
            basename = os.path.basename(raster_path).split('.')[0]
            raster_data = rasterio.open(raster_path)
            bounds = raster_data.bounds

            print(f'Processing: Extracting raster values from {basename} ...')
            geometry = [Point(xy) for xy in zip(training_sites_gee.longitude, training_sites_gee.latitude)]
            points_gdf = GeoDataFrame(training_sites_gee, geometry=geometry, crs='EPSG:4326')
            raster_crs = raster_data.crs
            points_gdf = points_gdf.to_crs(raster_crs)
            points_within_extent = points_gdf.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]

            n = 1
            for index, point in points_within_extent.iterrows():
                # print(f'Progress: [{basename}: {n}/{len(total_points)}] processed... ', end='\r')
                x, y = point.geometry.coords[0]
                row, col = raster_data.index(x, y)

                pixel_value = raster_data.read(1, window=((row, row+1), (col, col+1)))

                extracted_values = {
                    'longitude': point['longitude'],
                    'latitude': point['latitude'],
                    'land_cover_class': point['land_cover_class'],
                    'LCT': point['LCT']
                }
                # Add the extracted raster value to the corresponding dictionary
                basename = os.path.basename(raster_path).split('.')[0]
                extracted_values[f'{basename}'] = pixel_value[0, 0] if pixel_value is not None else None

                raster_values_dict[basename].append(extracted_values)
                n += 1

        # Convert the list of dictionaries for each raster path into DataFrames
        raster_dfs = {raster_path: pd.DataFrame(values) for raster_path, values in raster_values_dict.items()}

        # Create a base DataFrame with common columns
        common_cols = ['longitude', 'latitude', 'land_cover_class', 'LCT']
        training_sites_unclean = pd.DataFrame(training_sites_gee, columns=common_cols)

        # Merge the DataFrames for different raster paths
        for raster_path, raster_df in raster_dfs.items():
            training_sites_unclean = training_sites_unclean.merge(raster_df, on=common_cols, how='left')

        training_sites_unclean = training_sites_df_unclean.dropna()
        training_sites_unclean['LCT'] = training_sites_unclean['LCT'].astype('category')
        raster_value_columns = [f"{os.path.basename(raster_path).split('.')[0]}" for raster_path in raster_paths]

        print('Processing: Removing outliers (number of training sites may decrease) ...')
        z_scores = np.abs(stats.zscore(training_sites_unclean[raster_value_columns]))
        # Use a threshold for Z-score to consider values as outliers (e.g., Z-score > 3)
        outlier_rows = np.any(z_scores > z_score_threshold, axis=1)
        training_sites = training_sites_unclean[~outlier_rows]

        # Save the training_sites
        training_sites.to_csv(output_csv, index=False)

        return training_sites

def validate_results(test_df, classified_column_name='classified_LCT', reference_column_name='reference_LCT', opf=os.getcwd()):
    """
    Validate the results of a land cover classification by computing evaluation metrics.

    This function calculates evaluation metrics for a land cover classification based on the provided
    test DataFrame containing reference and classified land cover labels.

    Parameters:
        test_df (pandas.DataFrame): A DataFrame containing reference and classified land cover labels.
        classified_column_name (str): The name of the column containing the classified land cover labels.
        reference_column_name (str): The name of the column containing the reference land cover labels.
        opf (str): The output folder where the classification metrics and confusion matrix CSV files will be saved.

    Returns:
        dict: A dictionary containing the following classification metrics:
            - 'Overall Accuracy': The overall accuracy of the classification.
            - 'Producer Accuracy': A list of producer accuracy values for each land cover class.
            - 'User Accuracy': A list of user accuracy values for each land cover class.
            - 'Kappa': The Cohen's Kappa coefficient representing agreement between reference and classified labels.

    Example:
        test_data = {'reference_LCT': ['A', 'B', 'C', 'A', 'B', 'C'],
                     'classified_LCT': ['A', 'B', 'C', 'A', 'B', 'D']}  # Example test data
        test_df = pd.DataFrame(test_data)
        metrics = validate_results(test_df, classified_column_name='classified_LCT',
                                   reference_column_name='reference_LCT', opf='output_folder/')
        print(metrics)
    """

    # Extract the reference and classified columns
    reference = test_df[reference_column_name]
    classified = test_df[classified_column_name]
    class_names = np.unique(reference)

    # Create a confusion matrix
    cm = confusion_matrix(reference, classified)
    overall_accuracy = accuracy_score(reference, classified)

    producer_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    user_accuracy = np.diag(cm) / np.sum(cm, axis=0)
    kappa = cohen_kappa_score(reference, classified)

    metrics = {
        'Overall Accuracy': overall_accuracy,
        'Producer Accuracy': producer_accuracy,
        'User Accuracy': user_accuracy,
        'Kappa': kappa}

    class_names = list(class_names)

    # Separate user and producer accuracy
    user_accuracy = metrics['User Accuracy']
    producer_accuracy = metrics['Producer Accuracy']

    with open(os.path.join(opf, 'classification_metrics.txt'), 'w') as f:
        for metric, value in metrics.items():
            if isinstance(value, list):
                # Handle lists (User and Producer accuracy)
                f.write(f"{metric}:\n")
                for class_name, acc_value in zip(class_names, value):
                    f.write(f"  {class_name}: {acc_value:.3f}\n")
            elif isinstance(value, np.ndarray) and value.size > 1:
                # Handle arrays (e.g., User and Producer accuracy)
                f.write(f"{metric}:\n")
                for class_name, acc_value in zip(class_names, value):
                    f.write(f"  {class_name}: {acc_value:.3f}\n")
            else:
                # Handle other metrics
                scalar_value = value.item() if isinstance(value, np.ndarray) else value
                f.write(f"{metric}: {scalar_value:.3f}\n")

    cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
    cm_df.to_csv(os.path.join(opf, 'confusion_matrix.csv'))

    return metrics

def test_model(classified_path, test_points, opf):
    """
    Test a land cover classification model using reference points and a classified raster.

    This function tests a land cover classification model by comparing the classified raster with
    reference points. It calculates classification metrics and creates a DataFrame with test results.

    Parameters:
        classified_path (str): The file path to the classified raster.
        test_points (GeoDataFrame): A GeoDataFrame containing reference points with land cover labels.
        opf (str): The output folder where classification metrics and test results will be saved.

    Returns:
        tuple: A tuple containing two elements:
            - test_df (pandas.DataFrame): A DataFrame with test results, including reference and classified labels.
            - metrics (dict): A dictionary containing classification metrics, such as overall accuracy,
              producer accuracy, user accuracy, and Cohen's Kappa.

    Example:
        classified_raster = "classified.tif"
        test_reference_points = GeoDataFrame(...)  # Create a GeoDataFrame with reference points and labels
        output_folder = "output/"
        test_results, classification_metrics = test_model(classified_raster, test_reference_points, output_folder)
        print(classification_metrics)
    """

    # Open the classified raster
    raster_data = rasterio.open(classified_path)
    
    # Create a GeoDataFrame from test_points
    geometry = [Point(xy) for xy in zip(test_points.longitude, test_points.latitude)]
    points_gdf = GeoDataFrame(test_points, geometry=geometry, crs='EPSG:4326')
    raster_crs = raster_data.crs
    points_gdf = points_gdf.to_crs(raster_crs)
    
    test_val = []
    for index, point in points_gdf.iterrows():
        x, y = point.geometry.coords[0]
        row, col = raster_data.index(x, y)

        pixel_value = raster_data.read(1, window=((row, row+1), (col, col+1)))
        extracted_values = {'reference_LCT': point['land_cover_class'],
                            'classified_LCT': pixel_value[0][0]}
        test_val.append(extracted_values)

    test_df = pd.DataFrame(test_val)
    metrics = validate_results(test_df, 'classified_LCT', 'reference_LCT', opf)
    return test_df, metrics

def raster_to_df(raster_paths):
    """
    Convert a list of raster files to a pandas DataFrame.

    This function reads one or more raster files, flattens them into one-dimensional arrays,
    and creates a pandas DataFrame with each raster's pixel values as columns.

    Parameters:
        raster_paths (list): A list of file paths to the raster files to be converted.

    Returns:
        tuple: A tuple containing two elements:
            - df (pandas.DataFrame): A DataFrame with columns representing each raster's pixel values.
            - profile (dict): The profile of the first raster, including metadata.

    Example:
        raster_files = ["raster1.tif", "raster2.tif"]
        data_frame, raster_profile = raster_to_df(raster_files)
        print(data_frame.head())
    """
    
    df = pd.DataFrame()
    for raster_path in raster_paths:
        with rasterio.open(raster_path) as r:
            image = r.read(1)
            profile = r.profile
            column_name = f"{os.path.basename(raster_path).split('.')[0]}"
            df[column_name] = image.flatten()

    return df, profile

def rf_classifier(raster_paths, training_sites, opf, **kwargs):
    """
    Perform land cover classification using a Random Forest classifier.

    This function trains a Random Forest classifier on training sites' raster values and
    applies the trained classifier to classify land cover in raster images. The classified
    output is saved as a GeoTIFF file.

    Parameters:
        raster_paths (list): A list of file paths to the raster images for classification.
        training_sites (pandas.DataFrame): A DataFrame containing training sites' raster values
            and land cover class labels.
        opf (str): The output folder where the classified GeoTIFF file will be saved.
        **kwargs: Additional keyword arguments for configuring the Random Forest classifier,
            e.g., n_estimators, random_state, criterion, etc.

    Returns:
        tuple: A tuple containing two elements:
            - output_path (str): The file path to the saved classified GeoTIFF.
            - predicted_classes (numpy.ndarray): An array containing the predicted land cover classes
                for each pixel in the input raster.

    Example:
        raster_files = ["raster1.tif", "raster2.tif"]
        training_data = pd.read_csv("training_data.csv")
        output_file, predictions = rf_classifier(raster_files, training_data, "output_folder")
        print(f"Classification saved at: {output_file}")
    """
    
    # Set default values for n_estimators, random_state, and criterion
    n_estimators = kwargs.get('n_estimators', 100)
    random_state = kwargs.get('random_state', 0)
    criterion = kwargs.get('criterion', 'entropy')
    
    land_cover_classes = training_sites.loc[:, 'LCT']
    raster_value_columns = [f"{os.path.basename(raster_path).split('.')[0]}" for raster_path in raster_paths]
    raster_values = training_sites.loc[:, raster_value_columns]   
    
    print('Processing: Applying MinMaxScaling ...')
    scaler = MinMaxScaler()
    selected_columns = [col for col in training_sites.columns if col in raster_value_columns]
    training_sites[selected_columns] = scaler.fit_transform(training_sites[selected_columns])

    # Train the classifier
    print('Processing: Initializing Random Forest Classifier ...')
    classifier = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, criterion = criterion, **kwargs)
    classifier.fit(raster_values, land_cover_classes)

    # Open one of the rasters to get the image shape
    raster_data = rasterio.open(raster_paths[0])
    image_shape = (raster_data.height, raster_data.width)
    raster_df, profile = raster_to_df(raster_paths)

    # Run the prediction
    print('Processing: Running prediction ...')
    # predicted_classes = multi_output_classifier.predict(raster_df)
    predicted_classes = classifier.predict(raster_df)

    # Reshape predicted_classes to match original image shape
    predicted_classes = predicted_classes.reshape(image_shape[0], image_shape[1])
    class_mapping = [(predicted_classes == 'Tree cover'),
        (predicted_classes == 'Shrubland'),
        (predicted_classes == 'Grassland'),
        (predicted_classes == 'Cropland'),
        (predicted_classes == 'Built-up'),
        (predicted_classes == 'Bare / sparse vegetation'),
        (predicted_classes == 'Snow and ice'),
        (predicted_classes == 'Permanent water bodies'),
        (predicted_classes == 'Herbaceous wetland'),
        (predicted_classes == 'Mangroves'),
        (predicted_classes == 'Moss and lichen')]
    
    classes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    numeric_predicted_classes = np.select(class_mapping, classes, default=np.nan)

    output_path = os.path.join(opf, 'Classified.TIF')
    with rasterio.open(output_path, 'w', **profile) as output_raster:
        output_raster.write(numeric_predicted_classes, 1)

    return output_path, predicted_classes

def subset_test_points(train_points, max_test_points=20):
    """
    Subset a DataFrame of training points to generate a smaller test point set.

    This function takes a DataFrame of training points and subsets it to create a smaller
    test point set. The subset is created by randomly selecting a limited number of points
    from each class category in the training set.

    Parameters:
        train_points (pandas.DataFrame): A DataFrame containing training points with class labels.
        max_test_points (int): The maximum number of test points to be included in the subset.
            Default is 20.

    Returns:
        pandas.DataFrame: A DataFrame containing the subset of test points.

    Example:
        train_data = pd.read_csv("training_data.csv")
        test_data = subset_test_points(train_data, max_test_points=30)
        print(f"Generated {len(test_data)} test points.")
    """
    
    grouped = train_points.groupby('LCT')
    test_points = pd.DataFrame()
    max_points_per_category = max_test_points // len(grouped)

    for category, group in grouped:
        if len(group) <= max_points_per_category:
            sample_group = group.sample(n=len(group), random_state=42)
        else:
            sample_group = group.sample(n=max_points_per_category, random_state=42)
        test_points = pd.concat([test_points, sample_group])
        
    return test_points

def classify(raster_paths, training_points=100, uniform_to_random_ratio=0.5, train_to_test_ratio=0.8,
             window_size=1, z_score_threshold=3, force=False, **kwargs):
    """
    Perform land cover classification using Random Forest.

    This function carries out a land cover classification task using Random Forest. It generates a training set,
    subsets it for testing, trains a Random Forest classifier, and evaluates the classification results.

    Parameters:
        raster_paths (list): A list of file paths to raster images used for classification.
        training_points (int): The total number of training points to be generated. Default is 100.
        uniform_to_random_ratio (float): The ratio of uniform grid points to random points in the training set.
            Default is 0.5.
        train_to_test_ratio (float): The ratio of training points to testing points. Default is 0.8.
        window_size (int): The size of the window used for extracting land cover information around training points.
            Default is 1.
        z_score_threshold (int): The threshold value for removing outliers from the training set based on Z-scores.
            Default is 3.
        force (bool): If True, forces the re-calculation of training sites and overwrites existing data. Default is False.
        **kwargs: Additional keyword arguments passed to the Random Forest classifier.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: DataFrame of training sites.
            - pandas.DataFrame: DataFrame of test points.
            - numpy.ndarray: Array of predicted land cover classes.
            - dict: Metrics for evaluating the classification results.

    Example:
        raster_paths = ["image1.tif", "image2.tif"]
        train_data, test_data, predictions, metrics = classify(raster_paths, training_points=200, uniform_to_random_ratio=0.6)
        print(f"Overall Accuracy: {metrics['Overall Accuracy']:.3f}")
    """
    
    # classifier = train_ml_classifier(training_sites, threshold)
    opf = os.path.dirname(raster_paths[0])
    training_sites_df_unclean = generate_training_sites(raster_paths, training_points = training_points, 
                                                        uniform_to_random_ratio = uniform_to_random_ratio, 
                                                        window_size = window_size, 
                                                        z_score_threshold = z_score_threshold, 
                                                        force = force, 
                                                        opf = opf)

    train_points = training_sites.sample(frac = train_to_test_ratio, random_state = 42)
    test_points = subset_test_points(train_points, max_test_points = int(training_points * (1-train_to_test_ratio)))
    
    classified_path, predicted_classes = rf_classifier(raster_paths, train_points, opf, **kwargs)
    test_df, metrics = test_model(classified_path, test_points, opf)
    
    print('Processing: Completed.')
    return training_sites, test_df, predicted_classes, metrics

b5 = r'D:\Aerosol Modelling\Aerosol\Output\Landsat 8-9\Delhi_147-40_146-40\Delhi_26-10-2021_146-40_S\Resampled_SR\B5_20211026_resample.TIF'
b4 = r'D:\Aerosol Modelling\Aerosol\Output\Landsat 8-9\Delhi_147-40_146-40\Delhi_26-10-2021_146-40_S\Resampled_SR\B4_20211026_resample.TIF'
b3 = r'D:\Aerosol Modelling\Aerosol\Output\Landsat 8-9\Delhi_147-40_146-40\Delhi_26-10-2021_146-40_S\Resampled_SR\B3_20211026_resample.TIF'

raster_paths = [b3, b4, b5]
training_sites, test_df, predicted_image, metrics = classify(raster_paths, training_points = 350, uniform_to_random_ratio = 0.5, 
                                                             train_to_test_ratio = 0.8, window_size = 1, z_score_threshold = 2, force = True)