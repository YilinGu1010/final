# Unsupervised Mapping of Earthquake-Triggered Landslides from Sentinel-2 Imagery

This project employs unsupervised learning techniques, specifically K-means and Gaussian Mixture Models (GMM), on Sentinel-2 satellite imagery to identify landslide scars triggered by an earthquake. The infographic below outlines the overall workflow of the project.

![Image Alt](https://github.com/YilinGu1010/final/blob/d6fa01dc12fd7c3168871e64219250d37ccf885f/Screen%20Shot%202025-06-05%20at%2018.47.31.png)

## About the project

This project serves as the final assignment for GEOL0069: Artificial Intelligence for Earth Observation. It investigates the application of unsupervised learning techniques—namely K-means clustering and Gaussian Mixture Models (GMM)—to detect landslide scars using pre- and post-earthquake Sentinel-2 optical imagery. The full code implementation can be found in the notebook titled AI4EO_Final_Project.ipynb.

### Background

Landslides are destructive natural events frequently triggered by earthquakes or intense rainfall, posing serious risks to infrastructure and human life. Prompt and accurate identification of areas affected by landslides is essential for effective post-disaster response. Additionally, landslide scars can play a crucial role in informing hydrogeological models used to evaluate slope stability (Montgomery & Dietrich, 1994)(Karakas et al,. 2024)(Görüm et al,. 2023)(Heidarzadeh et al,. 2023). Satellite-based remote sensing provides a powerful means of monitoring vast or hard-to-access regions, with optical imagery offering valuable spectral data for detecting vegetation depletion and surface disruption. In particular, the Sentinel-2 satellite, known for its high-resolution multispectral capabilities, is particularly well-suited for mapping landslide events.

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

The Sentinel-2 images in this study were not perfectly aligned, especially between the before and after earthquake scenes. To ensure accurate pixel-by-pixel comparison, we corrected this misalignment using Enhanced Correlation Coefficient (ECC) alignment (Evangelidis & Psarakis, 2008), aligning the post-earthquake image to the pre-earthquake one. The result of this alignment is shown below:

![Image Alt](https://github.com/YilinGu1010/final/blob/d8ba6b55ff5b95f64a08a4c460962ce15844505c/pre.png)
*Here are the RGB images of the study area showing the pre-earthquake scene, the post-earthquake scene before alignment, and the post-earthquake scene after alignment.*

## Normalised Difference Vegetation Index (NDVI) Mask

Following the alignment of the pre- and post-earthquake images, a ground truth dataset was established by applying a conventional threshold-based approach using the Normalized Difference Vegetation Index (NDVI). Specifically, regions where the NDVI value dropped by more than 0.2—signifying significant vegetation loss—were detected and subsequently used to create a mask representing the landslide scars. This mask serves as a reference for evaluating the accuracy of landslide detection methods in the study.

![Image Alt](https://github.com/YilinGu1010/final/blob/d05a4c6ae3e18188a0b91a18c5a1f74ddde0cb0b/NDVI.png)
*The NDVI maps for both the pre-earthquake and the aligned post-earthquake images, along with the resulting NDVI-based mask.*

## Unsupervised Learning

Because the direct use of raw spectral band data for detecting landslide scars yielded poor results, this project sought alternative input features that could improve detection accuracy. Instead of relying on the original band values, which often have limited ability to distinguish between vegetation and bare soil, the study focused on indices that highlight relevant surface characteristics. Specifically, the Normalized Difference Vegetation Index (NDVI) was used to capture vegetation health and density, while the Bare Soil Index (BSI) was employed to emphasize exposed soil areas. By using these two indices as input features for the unsupervised learning models, the project aimed to avoid the limitations of traditional threshold-based methods that require prior knowledge or manual setting of cut-off values. This approach enables the detection of landslide scars in a more flexible and data-driven manner, leveraging spectral information that better differentiates land cover types critical for identifying areas affected by landslides.

### Bare Soil Index (BSI)

Bare soil generally has distinct spectral properties compared to vegetated surfaces, which can be leveraged for its identification using remote sensing data. Specifically, bare soil tends to reflect more light in the visible red and blue bands, making it appear brighter in these wavelengths compared to areas covered by healthy vegetation. Additionally, in the shortwave infrared (SWIR) region, bare soil usually exhibits a strong reflectance response, while in the near-infrared (NIR) region, its reflectance is relatively low. These contrasting reflectance patterns across different spectral bands provide a useful basis for distinguishing bare soil from vegetation. By combining these spectral characteristics—higher reflectance in red and blue bands, strong reflectance in SWIR, and lower reflectance in NIR—researchers can effectively detect and map exposed soil surfaces. This spectral behavior has been well documented in previous studies (Roy et al., 1996) and forms the foundation for indices like the Bare Soil Index (BSI), which enhances the ability to identify bare soil in satellite imagery. Utilizing such indices is particularly valuable for applications such as landslide detection, where changes in soil exposure are key indicators of surface disturbance.

### K-Means

K-means clustering is a widely used unsupervised machine learning algorithm designed to divide a dataset into a specified number of clusters, denoted by k, based on the similarity of data points' features (MacQueen, 1967). The process begins by initializing k centroids, which represent the center of each cluster. Then, each data point in the dataset is assigned to the cluster whose centroid is closest, typically measured using Euclidean distance. After all points are assigned, the centroids are recalculated as the mean position of all points within each cluster. This assignment and update process repeats iteratively until the centroids stabilize and the within-cluster variance—the sum of squared distances between data points and their cluster centroids—is minimized. The goal of K-means is to produce compact, well-separated clusters where data points within each cluster share high similarity, making it a popular choice for tasks such as image segmentation, pattern recognition, and in this project, distinguishing different land cover types like vegetation and bare soil.

In this project, K-means clustering was employed to distinguish landslide-affected areas from unaffected regions using NDVI and BSI indices calculated from Sentinel-2 images captured before and after the earthquake. Two different clustering scenarios were explored: one with two clusters and another with three clusters. The purpose of testing these scenarios was to evaluate if introducing an additional cluster could enhance the classification accuracy, especially by minimizing the errors where vegetation with variable reflectance might be mistakenly classified as bare soil.

![Image Alt](https://github.com/YilinGu1010/final/blob/20db79bb2a233bdb50793147564d901224bcf002/K-Mean2.png)
*Landslide scar detection using K-means clustering with two clusters.*
  
![Image Alt](https://github.com/YilinGu1010/final/blob/a150236ba28b781536ef4b747c6d6763640be6c0/K-Mean3.png)
*Landslide scar detection using K-means clustering with three clusters.*

### Gaussian Mixture Models (GMM)

Gaussian Mixture Models (GMMs) are a type of probabilistic model that assume the data is generated from a combination of several Gaussian (normal) distributions, each representing a subpopulation within the overall dataset. Each Gaussian component is defined by its own mean vector and covariance matrix, allowing the model to capture variations in data spread and orientation (Reynolds, 2009). Unlike simpler clustering algorithms such as K-means, which assign data points to hard clusters based solely on distance to centroids, GMMs perform soft clustering by estimating the probability that each data point belongs to each Gaussian component. This flexibility allows GMMs to model clusters with different shapes, sizes, and degrees of overlap, making them well-suited for complex datasets with mixed or overlapping classes.

In this project, GMMs were used to classify landslide scars by clustering NDVI and Bare Soil Index (BSI) values derived from Sentinel-2 satellite imagery collected before and after the earthquake event. The model was tested with two different numbers of components — two and three — to investigate whether increasing the number of mixture components could improve the classification accuracy. By allowing more components, the model may better capture subtle variations in the spectral indices caused by vegetation changes and soil exposure, thus enhancing the distinction between landslide-affected and unaffected areas. This approach aimed to overcome the limitations of traditional threshold-based methods by providing a more flexible, data-driven means of identifying landslide scars without requiring prior knowledge of the local environment.

![Image Alt](https://github.com/YilinGu1010/final/blob/7c91a5823163dd006b9996f215664c02a406a301/GMM2.png)
*Landslide scar detection using Gaussian Mixture Model (GMM) clustering with two components.*

![Image Alt](https://github.com/YilinGu1010/final/blob/62b6be193736805d2c373fa807e1cf990d4964ce/GMM3.png)
*Landslide scar detection using Gaussian Mixture Model (GMM) clustering with three components.*

## Performance Analysis

To evaluate the effectiveness of the four different models used in this study, the landslide mask generated from the NDVI threshold method was employed as the ground truth or reference standard. This mask represents areas identified as landslide scars based on a significant decrease in vegetation index, providing a baseline against which the performance of the clustering models—both K-means and Gaussian Mixture Model (GMM)—could be compared.

The evaluation process began by calculating confusion matrices for each model’s classification results. These matrices summarize how well the predicted labels align with the reference, showing true positives, true negatives, false positives, and false negatives. From these values, various performance metrics such as overall accuracy, precision, recall, and F1-score were derived to give a quantitative assessment of each method’s capability to correctly identify landslide-affected areas while minimizing misclassification.

Additionally, classification reports generated by the code offered detailed insights into each model’s performance, highlighting strengths and weaknesses in detecting the landslide scars. This comprehensive analysis helps to understand not just the raw accuracy but also the balance between correctly identifying landslides and avoiding false alarms.

To complement these quantitative evaluations, all predicted landslide masks were visually compared by overlaying them on the post-earthquake true-colour satellite imagery of the study area. This visual comparison provides an intuitive way to assess how well the models capture the spatial patterns of landslide scars and whether the predicted areas correspond with visible changes in the landscape. Together, these analyses enable a thorough assessment of the models’ reliability and practical utility for landslide detection using remote sensing data.

![Image Alt](https://github.com/YilinGu1010/final/blob/3cb64871fc0320deae76c8512848dd35cbcb77d3/Summary.png)
*Confusion matrices for each model*

![Image Alt](https://github.com/YilinGu1010/final/blob/d8aaa5a19c4718899e570bcd4b20a6e351b21061/six.png)
*Post-earthquake true-color images accompanied by classification overlays are presented in six panels: (1) the original post-earthquake true-color image, (2) the landslide mask derived from NDVI, (3) results of K-means clustering with two clusters, (4) K-means clustering using three clusters, (5) Gaussian Mixture Model (GMM) clustering with two components, and (6) GMM clustering with three components.*

## Conclusion

The results show that using three groups in K-means and GMM helps better tell apart vegetation and bare soil. However, when it comes to finding landslide scars by comparing images from before and after the earthquake, all the methods work pretty well and give similar results. This means that unsupervised learning is a useful and efficient way to detect landslide scars without needing detailed prior knowledge about the local plants. So, depending on what you want to do, you can choose the number of groups: for spotting landslides over time, two groups are enough, but for better separating soil and plants in a single image, using three groups works better.

## References

Kocaman, S., Çetinkaya, S., TUNAR ÖZCAN, N.A.Z.L.I., Karakaş, G., Karakaş, V.E. and Gökçeoğlu, C., 2025. Landslides triggered by the 6 February 2023 Kahramanmaraş earthquakes (Türkiye). Turkish Journal of Earth Sciences, 34(1), pp.47-67.

MacQueen, J., 1967, January. Some methods for classification and analysis of multivariate observations. In Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Statistics (Vol. 5, pp. 281-298). University of California press.

Wang, Y., Su, J., Zhai, X., Meng, F. and Liu, C., 2022. Snow coverage mapping by learning from sentinel-2 satellite multispectral images via machine learning algorithms. Remote Sensing, 14(3), p.782.

Montgomery, D.R. and Dietrich, W.E., 1994. A physically based model for the topographic control on shallow landsliding. Water resources research, 30(4), pp.1153-1171.

Ding, Z. and Wang, C., 2025. Coseismic landslides caused by the 2022 Luding earthquake in China: insights from remote sensing interpretations and machine learning models. Frontiers in Earth Science, 13, p.1564744.

KRIEGLER, F.J., 1969. Preprocessing transformations and their effects on multspectral recognition. In Proceedings of the sixth international symposium on remote sesning of environment (pp. 97-131).

Saito, H., Uchiyama, S. and Teshirogi, K., 2022. Rapid vegetation recovery at landslide scars detected by multitemporal high-resolution satellite imagery at Aso volcano, Japan. Geomorphology, 398, p.107989.

Yunus, A.P., Fan, X., Tang, X., Jie, D., Xu, Q. and Huang, R., 2020. Decadal vegetation succession from MODIS reveals the spatio-temporal evolution of post-seismic landsliding after the 2008 Wenchuan earthquake. Remote Sensing of Environment, 236, p.111476.

Yang, W., Wang, Y., Sun, S., Wang, Y. and Ma, C., 2019. Using Sentinel-2 time series to detect slope movement before the Jinsha River landslide. Landslides, 16, pp.1313-1324.

Karakas, G., Unal, E.O., Cetinkaya, S., Ozcan, N.T., Karakas, V.E., Can, R., Gokceoglu, C. and Kocaman, S., 2024. Analysis of landslide susceptibility prediction accuracy with an event-based inventory: The 6 February 2023 Turkiye earthquakes. Soil Dynamics and Earthquake Engineering, 178, p.108491.

Görüm, T., Tanyas, H., Karabacak, F., Yılmaz, A., Girgin, S., Allstadt, K.E., Süzen, M.L. and Burgi, P., 2023. Preliminary documentation of coseismic ground failure triggered by the February 6, 2023 Türkiye earthquake sequence. Engineering Geology, 327, p.107315.

Heidarzadeh, M., Gusman, A.R. and Mulia, I.E., 2023. The landslide source of the eastern Mediterranean tsunami on 6 February 2023 following the Mw 7.8 Kahramanmaraş (Türkiye) inland earthquake. Geoscience letters, 10(1), p.50.

## Video Tutorial






