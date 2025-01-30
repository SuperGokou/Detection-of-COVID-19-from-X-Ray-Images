Detection of COVID-19 from X-Rays

Group #7: Ming Xia, Herman Tjahjadi, Jacqueline Jiang, Jinyu Han, Paul Li

Description of the Dataset
The first dataset we used, the “Brixia score COVID-19 Dataset”, was collected during the pandemic's peak (March 4th to April 4th, 2020) at the ASST Spedali Civili di Brescia. It encompasses a large set of Chest X-ray (CXR) images used for triage and patient monitoring in sub-intensive and intensive care units, reflecting the variability of an actual clinical scenario. The dataset comprises 4,695 CXR images of COVID-19 subjects, captured using both CR and DX modalities in AP or PA projections, sourced from the facility’s RIS-PACS system. All images in this dataset represent positive cases and are intended to serve as positive examples for training purposes.

The second dataset we used is the “NIH Chest X-ray Dataset,” a large-scale public dataset from the National Institutes of Health (NIH). This collection consists of 112,120 frontal-view X-ray images of 30,805 unique patients. Each image in the dataset is annotated with one or more of 14 different thoracic pathology labels, or "No Finding," for images that do not exhibit any of the 14 pathologies. To train a model on COVID-19 detection, we would specifically focus on utilizing the images labeled "No Finding" as negative instances. 

The last dataset we used is the "CovidGR Dataset," which is a collection specifically aimed at fostering the development and evaluation of machine-learning models for COVID-19 detection from chest X-ray images. This particular repository is managed by the research group "ari-dasci" and focuses on providing open datasets and resources for tackling the challenge of automatic COVID-19 diagnosis using imaging data. All the data images serve as a test set and provide external validation.


Quality and Resolution of Images
The diagnostic quality of X-ray images may vary, affecting the model's ability to learn from them. Variations in resolution, contrast, and noise levels can introduce challenges in identifying relevant features for COVID-19.
Aligning the datasets with different data formats. The images from Brixia are 3D DCM image format while the images from NIH are 2D images in PNG. Finally, the images from the CovidGR dataset are images in JPG. We observed that the images have different sizes and the chest XRays show strong augmentation with various twists and orientations of the spines.

Explain how these issues have been or will be addressed before the next milestone.

Addressing Quality and Resolution of Images

Standardization and Preprocessing: Implement image preprocessing techniques to normalize the images across the datasets. This includes adjusting the image size, enhancing contrast, and applying noise reduction techniques to mitigate variations in image quality and imaging conditions.
Data Augmentation: Data augmentation techniques such as rotation, flipping, scaling, and cropping can be employed to combat overfitting and improve the model's ability to generalize across different imaging conditions. This increases the dataset's effective size and introduces variability that helps the model learn more robust features.


Addressing Data Imbalance

Adding Positive COVID-19 X-ray Samples Data: To address the issue of data imbalance due to the limited number of positive COVID-19 X-ray samples, which stands at 4,695, it's essential to consider methods for augmenting the dataset. Acquiring additional positive X-ray samples would be beneficial for improving the model's accuracy. 

Class Weights:

Utilize the computed class weights in the training process to adjust the model's focus, penalizing the misclassification of the minority class more than the majority class to counteract the imbalance.
Oversampling and Undersampling: Experiment with oversampling the minority class or undersampling the majority class to create a more balanced training dataset. Techniques like SMOTE (Synthetic Minority Over-sampling Technique) can generate synthetic examples of the minority class to enhance balance.
