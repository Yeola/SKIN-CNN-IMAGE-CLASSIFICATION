# Skin Image Classification

## Table Of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset Sources](#dataset-sources)
- [Tools and Technologies](#tools-and-technologies)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Model Performance](#model-performance)
- [Model Interpretation and insights](#model-interpretation-and-insights)
- [Areas For Improvement](#areas-for-improvement)
- [Bias and Fairness Analysis](#bias-and-fairness-analysis)
- [Mitigation Strategies](#mitigation-strategies)
### Project Overview

This project focuses on building a Convolutional Neural Network (CNN) to classify skin conditions from images. The model predicts whether an image shows:
- Vitiligo
- Acne
- Hyperpigmentation.
  
  The goal is to assist in early detection and awareness of skin conditions using deep learning.

### Objectives

1. Build a custom CNN model for image classification.
2. Improve model performance using data augmentation.
3. Evaluate performance using accuracy, loss, confusion matrix, and classification report.
4. Prepare the model for future deployment as a web application.

### Dataset Sources&Description

The primary dataset used for this analysis is the "skin_dataset" containing images about all the categories of skin conditions selected. skin_dataset was gotten by merging images scraped from dermnet website to an already existing skin_diseases_dataset. The dataset consists of labeled skin images categorized into:
- Vitiligo
- Acne
- Hyperpigmentation

### Tools and Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Jupyter Notebook

### Data Cleaning and Preparation

In the initial data preparation phase, we performed the following tasks:
- Data loading and inspection.
- Cleaning (duplicates removed)
- Renaming systematically
- Resizing to 50×50
- Split into 80% training and 20% testing

### Model Architecture

```python
input_shape=(50,50,3)
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(3, activation='softmax'))
```

### Training Configuration

- Loss Function: categorical_crossentropy
- Optimizer: Adam
- Metrics: accuracy
- Epochs: Longer training(100+ epochs)
- Batch size: 32
- Data Augmentation:
    - Rotation
    - Width & Height Shift
    - Zoom
    - Horizontal Flip

### Model Performance

- Test Accuracy: 93%
- Macro Avg F1-score: 0.92
- Weighted Avg F1-score: 0.93
- 
#### Classification Report

|                 |Precision|Recall  |f1-score|Support|
|-----------------|---------|--------|--------|-------|
|Vitiligo         |    0.96 |    0.94|   0.95 |  374  |
|Acne             |   0.92  |    0.94|   0.93 |   326 |
|Hyperpigmentation|  0.86   |  0.87  |   0.87 |   144 |
|                 |         |        |        |       |
|accuracy         |         |        |  0.93  |  844  |
|macro avg        | 0.92    | 0.92   | 0.92   | 844   |       
|weighted avg     | 0.93    |  0.93  |  0.93  |  844  |

#### Confusion Matrix

[[353    12     9]

 [8      307    11]
 
 [6      13     125]]

### Model Interpretation and Insights

#### What the model learned well:
- The model shows excellent performance on Vitiligo and Acne, with precision and recall above 92%.
- It demonstrates strong feature extraction ability. Despite its simplicity, the CNN successfully learned:
  - Texture Patterns
  - Pigmentation contrast
  - Skin lesion boundaries 
- The balanced precision and recall values indicate stable generalization, not just memorization.
- The longer training helped the model converge without needing deeper layers

### Areas For Improvement

- Hyperpigmentation has slightly lower performance (F1 = 0.87), likely due to:
   - Fewer samples (class imbalance).
   - Higher visual similarity with other skin conditions.

 ### Bias and Fairness Analysis

The dataset shows class imbalance, especially in hyperpigmentation, which may cause:
- Lower confidence for minority classes.
- Potential prediction bias toward majority classes.

The model may perform better on image styles similar to the training set and less reliably on:
- Different lighting conditions
- Different skin tones not well represented
- Images from different devices or environments

#### Mitigation strategies:

- Collect more hyperpigmentation samples.
- Add skin-tone-balanced data.
- Use class weights or focal loss in future training.

