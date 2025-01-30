# COVID-19 Detection from Chest X-Rays

## Team Members
- Ming Xia
- Herman Tjahjadi
- Jacqueline Jiang
- Jinyu Han
- Paul Li

## Project Description
This project aims to detect COVID-19 from chest X-ray images using machine learning models. We utilize three datasets with varying image qualities and diagnoses to train our models for accurate and efficient COVID-19 detection.

### Datasets

1. **Brixia Score COVID-19 Dataset**
   - **Source:** ASST Spedali Civili di Brescia
   - **Period:** March 4th to April 4th, 2020
   - **Content:** 4,695 CXR images of COVID-19 subjects, captured using CR and DX modalities in AP or PA projections. Images are sourced from the facility’s RIS-PACS system and represent positive cases.

2. **NIH Chest X-ray Dataset**
   - **Source:** National Institutes of Health (NIH)
   - **Content:** 112,120 frontal-view X-ray images of 30,805 unique patients, annotated with one or more of 14 different thoracic pathology labels or "No Finding" for images without any of the 14 pathologies.

3. **CovidGR Dataset**
   - **Source:** ari-dasci research group
   - **Purpose:** Provides a test set for external validation of machine-learning models focused on COVID-19 detection.

### Issues with Datasets

#### Quality and Resolution of Images
The diagnostic quality of X-ray images varies, presenting challenges in training models effectively. We observe differences in resolution, contrast, and noise levels across datasets which can hinder feature identification essential for COVID-19 detection.

#### Data Formats and Alignment
The images from Brixia are in 3D DCM format, whereas the images from NIH are 2D PNGs, and those from CovidGR are in JPG format. There are also noticeable size differences and orientations in the chest X-rays.

## Solutions and Preprocessing Steps

### Standardization and Preprocessing
To ensure consistency across datasets, we apply the following preprocessing techniques:
```python
# Image size adjustment
# Contrast enhancement
# Noise reduction
