# Unsupervised Mapping of Earthquake-Triggered Landslides from Sentinel-2 Imagery

This project employs unsupervised learning techniques, specifically K-means and Gaussian Mixture Models (GMM), on Sentinel-2 satellite imagery to identify landslide scars triggered by an earthquake. The infographic below outlines the overall workflow of the project.

![Image Alt](https://github.com/YilinGu1010/final/blob/d6fa01dc12fd7c3168871e64219250d37ccf885f/Screen%20Shot%202025-06-05%20at%2018.47.31.png)

## About the project

This project serves as the final assignment for GEOL0069: Artificial Intelligence for Earth Observation. It investigates the application of unsupervised learning techniques—namely K-means clustering and Gaussian Mixture Models (GMM)—to detect landslide scars using pre- and post-earthquake Sentinel-2 optical imagery. The full code implementation can be found in the notebook titled AI4EO_Final_Project.ipynb.

### Background

Landslides are destructive natural events frequently triggered by earthquakes or intense rainfall, posing serious risks to infrastructure and human life. Prompt and accurate identification of areas affected by landslides is essential for effective post-disaster response. Additionally, landslide scars can play a crucial role in informing hydrogeological models used to evaluate slope stability (Montgomery & Dietrich, 1994). Satellite-based remote sensing provides a powerful means of monitoring vast or hard-to-access regions, with optical imagery offering valuable spectral data for detecting vegetation depletion and surface disruption. In particular, the Sentinel-2 satellite, known for its high-resolution multispectral capabilities, is particularly well-suited for mapping landslide events.

On February 6, 2023, a magnitude 7.8 earthquake struck the border region between Turkey and Syria, triggering over 2,596 landslides (Kocaman, Sultan, et al., 2025). Landslides can lead to secondary disasters, causing some great damages. Assessing landslide risk in the surrounding areas is therefore of critical importance. This project focuses on the epicentral region, where landslides are densely concentrated. Moreover, the selected area features very low cloud cover in both pre- and post-earthquake Sentinel-2 imagery, providing favorable conditions for conducting this analysis.

One of the most commonly employed techniques for detecting landslide scars using remote sensing is the Normalised Difference Vegetation Index (NDVI) (Kriegler et al,. 1969). Because healthy vegetation reflects strongly in the near-infrared spectrum while absorbing red light, the NDVI leverages these spectral differences to estimate vegetation density. By comparing NDVI values from before and after the earthquake, it is possible to preliminarily locate areas impacted by landslides (e.g., Saito et al., 2022; Yunus et al., 2020). However, NDVI-based methods often rely on predefined thresholds, which depend on prior understanding of local vegetation conditions (Yang et al., 2019). To overcome this constraint, this study explores the use of unsupervised learning—specifically K-means clustering and Gaussian Mixture Models—to investigate the feasibility of detecting landslide scars without requiring any pre-existing knowledge about the local environment.

### Study Area & Date

In this project, I selected the area between longitudes 36.9450 and 38.3000, and latitudes 36.2020 and 37.3500. To minimise seasonal variations in vegetation and ensure low cloud cover, Sentinel-2 imagery from January 5 and August 5, 2023—representing pre- and post-earthquake conditions—was chosen for comparison.

## Getting Started

This project was carried out using Google Colab, which offers free GPU access and convenient integration with Google Drive for data storage. Alternatively, it can be executed in a local environment, provided that all required packages are installed and adequate computational resources are available.

### Prerequisite

Before starting the project in Google Colab, the rasterio package needs to be installed to enable reading, writing, and analysis of geospatial raster data. Additionally, to estimate the project's carbon emissions, the codecarbon package must also be installed.

```python
!pip install rasterio
!pip install codecarbon
```

After installation, these packages should be imported together with the other necessary libraries. If running the project locally, make sure all dependencies are correctly installed.

### Sentinel-2 Data

Sentinel-2 is a satellite mission launched by the European Space Agency (ESA) as part of the Copernicus program, aimed at delivering high-resolution optical imagery for land monitoring purposes. It collects data in 13 spectral bands spanning from visible light to shortwave infrared, with spatial resolutions of 10, 20, or 60 meters depending on the specific band. This research utilizes Level-2A (L2A) products, which have undergone atmospheric correction to provide Bottom-Of-Atmosphere (BOA) reflectance, derived from Level-1C Top-Of-Atmosphere (TOA) images (European Space Agency, date not specified). Because the shortwave infrared (SWIR) bands needed in this study are unavailable at the 10-meter resolution, all spectral bands are processed at 20-meter resolution to maintain uniformity. To fetch the data, you must create an account on the Copernicus Open Access Hub and input your login credentials in the "Fetching Data" section of the code.

```python
# Authenticate with Copernicus Data Space
username = "zcfbgub@ucl.ac.uk" # Change to your username
password = "20030526Gyl!" # Change to your password
access_token, refresh_token = get_access_and_refresh_token(username, password)

# Define Time Ranges for Pre- and Post-earthquake
pre_eq_start_date = "2023-01-05"
pre_eq_end_date = "2023-01-06"
post_eq_start_date = "2023-08-05"
post_eq_end_date = "2023-08-06"

# Query Sentinel-2 Products Covering 2023 TurkeySyria Earthquake Affected Area
pre_eq_sentinel2_data = query_sentinel2_TurkeySyria_data(
    pre_eq_start_date, pre_eq_end_date, access_token
)

post_eq_sentinel2_data = query_sentinel2_TurkeySyria_data(
    post_eq_start_date, post_eq_end_date, access_token
)

# Download the Selected Sentinel-2 Product for Each Time Range
download_dir = "/content/drive/MyDrive/AI/Week 10/"

# Download Pre-Earthquake Product
product_id = pre_eq_sentinel2_data['Id'][0]
file_name = pre_eq_sentinel2_data['Name'][0]
download_single_product(product_id, file_name, access_token, download_dir)

# Download Post-Earthquake Product
product_id = post_eq_sentinel2_data['Id'][0]
file_name = post_eq_sentinel2_data['Name'][0]
download_single_product(product_id, file_name, access_token, download_dir)
```

We can use the code above to download the file. The downloaded file will be in ZIP format. To access the data, unzip the file in the directory where it was saved.

## Data Alignment

## Normalised Difference Vegetation Index (NDVI) Mask

## Unsupervised Learning

### Bare Soil Index (BSI)

### K-Means

### Gaussian Mixture Models (GMM)

## Performance Analysis

## Conclusion

## Video Tutorial

## References
