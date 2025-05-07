# MILESTONE 1

This milestone is intended to apply unsupervised techniques to segment pulmonary lesions in volumes of interest (VOIs) extracted from CT scans. This involves implementing a standard pipeline over intensity volumes, including the use of Otsu thresholding combined with various morphological operations to refine segmentation results. Additionally, k-means clustering will be applied to classic filter banks to explore alternative segmentation strategies. The performance of these methods will be quantified using standard segmentation metrics, and a comparative analysis will be conducted to discuss their respective advantages, disadvantages, and sensitivity to parameters.

The repository is organized as follows:

- Data: Contains all the input data, including full datasets and samples used to create the annotations, segmentations, and results.

- Output: Contains all the results of this project, including the annotated dataset, VOIs of the sample data, and masks generated from the segmentation of the entire dataset.

- Codes: Includes the definitive scripts used to perform each task and save the results in the output folder.