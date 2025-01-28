# Instance Segmentation on Cityscapes dataset

## Requirements

Git with readme

### Main requirements:
 - Preparation and presentation of a selected data set,
 - Preparation and presentation of the main functionality that should be based on neural networks,
 - Presentation of the report,
 - The final grade depends on the number of points scored for each element of the project

### Dataset:
 - Minimum input size: **200x200px**
 - At least 1000 photos

### Training:
 - Correctly selected loss function
 - Data split into train, validation and test
 - Metric (at least 2) (?)

### Report:
 - description of the data set, with a few image examples
 - description of the problem
 - description of used architectures with diagram showing the layers; For large models containing blocks, the blocks and the connections between them can be shown separately.
 - model analysis: size in memory, number of parameters,  
 - description of the training and the required commands to run it
 - description of used metrics, loss, and evaluation
 - plots: training and validation loss, metrics
 - used hyperparameters along with an explanation of each why such value was chosen
 - comparison of models
 - list of libraries and tools used can be a requirements.txt file
 - a description of the runtime environment
 - training and inference time,
 - preparation of a bibliography - the bibliography should contain references to the data set (preferably the article in which the collection was presented) and all scientific works and studies, including websites with tips on the solution.
 - a table containing all the completed items with points.
 - link to Git

## Goals table:

| **Task** | **Points** | **Done** |
| --- | --- | --- |
| **Problem** | At least 1p | X |
| Instance Segmentation | 3 | X |
| Additional loss functions to improve prediction quality | 1 | X |
| **Model** | At least 3p | Soon|
| Maybe our own model (?) | 2 | Soon |
| architecture from the internet trained from scratch (mask_rcnn) | 1 | X |
| **Dataset** |
| Your own part of the dataset (>500 photos) | 1 | Soon |
| **Training** |
| Data augmentation | 1 | X |
| **Additional points** |
| Tensorboard | 1 | X |
| Streamlit | 1 | X |
| DVC | 2 | X |
| **Total** | **14** | **10** |
