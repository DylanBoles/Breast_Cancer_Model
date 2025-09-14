# Breast Cancer Pathology Classification
## Project Overview

This project focuses on classifying digital pathology images of breast tissue using both Deep Learning and Machine Learning methods. The dataset consists of 2D Discrete Cosine Transform (DCT) coefficients extracted from annotated pathology images.

The main objective was to develop models that could classify images into one of nine categories, with particular emphasis on six “interesting” classes relevant for cancer diagnosis.

I implemented two approaches:
- A Convolutional Neural Network (CNN) trained on inverse-DCT reconstructed images.
- A Random Forest classifier trained directly on the DCT coefficients.

The models were evaluated in a class competition based on their performance on a blind evaluation set, with special scoring focused on cancer-related labels. I placed 2nd overall in the competition.

## Dataset

The dataset was derived from v4.0.0 of the TUH DPATH Breast dataset. Each annotated patch underwent a 256×256 DCT, and the top 32×32 coefficients were retained for each channel (R, G, B).

Input format: CSV files with rows of [label, 3072 DCT coefficients]

Labels (0–8):

0 = norm
1 = artf
2 = nneo
3 = infl
4 = susp
5 = dcis
6 = indc
7 = null
8 = bckg

For scoring, labels were collapsed into:
- Ignored: artf (1), susp (4), null (7)
- Interesting (focus): norm (0), nneo (2), infl (3), dcis (5), indc (6), bckg (8)

## Methods
### Convolutional Neural Network (CNN)
- Reconstructed RGB images from DCT coefficients using inverse DCT.
- Applied data augmentation (rotation, flips, affine transforms).
#### Architecture:
- 4 convolutional blocks (Conv → BatchNorm → ReLU → MaxPool → Dropout)
- Fully connected layer with dropout
- Output: 9-class softmax
Loss: Class-weighted CrossEntropy
Optimizer: Adam with learning rate scheduling

### Random Forest Classifier
- Worked directly with the 3072-D feature vectors.
- Standardized features using StandardScaler.
#### Hyperparameters tuned for generalization:
- 800 trees
- max_depth=12
- class_weight="balanced"
- max_features="sqrt"

### Visualization
To better understand the dataset and present results, I implemented code to reconstruct and display pathology images from DCT coefficients, grouped by label class.

## Results
#### CNN Performance
- Strengths: Learned feature representations, balanced predictions across classes.
- Weaknesses: Overfitting persisted; dev/eval accuracy lagged.
- Final Score on Eval: ~38.8%

#### Random Forest Performance
- Strengths: Simpler, robust model with strong generalization.
- Weaknesses: Less interpretability in decision space.
- Final Score on Eval: ~54.1%

#### Competition Placement
- My Random Forest model achieved the 2nd highest score in the class competition.
- CNN performance was competitive but less effective than Random Forest on the blind evaluation set.

## Key Takeaways
- Class imbalance and overfitting were major challenges.
- Classical ML (Random Forests) outperformed deep learning (CNN) for this dataset due to limited training samples and strong regularization needs.
- Visualization of reconstructed images helped bridge the gap between raw coefficients and interpretability.
