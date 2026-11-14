# Chest X-Ray Pneumonia Classification

An educational deep learning project classifying chest X-ray
images as NORMAL or PNEUMONIA.

This project explores CNNs, data augmentation, transfer learning,
and the importance of checking generalization beyond validation
accuracy.

## Dataset

[Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

The selected dataset folders contained:
- Training: 5,216 images
- Original validation: 16 images
- Original test: 624 images

The original validation folder was not used. Instead, the training
folder was split into training and validation sets using inferred
filename groups.

The dataset is not included in this repository.

## Tools

Python, TensorFlow/Keras, Pandas, NumPy, Matplotlib,
and scikit-learn. The notebook was developed on Kaggle.

## Workflow

1. Inspect folder structure, image counts, and sample images.
2. Organize image paths and labels in a DataFrame.
3. Infer groups from filenames and create a grouped,
   class-stratified training/validation split.
4. Resize images to 224 × 224, load three channels, and normalize
   pixels to [0, 1].
5. Train and compare four models using early stopping.
6. Select a model using validation results.
7. Inspect development/test filename-group overlap and exact
   file duplicates.
8. Evaluate the selected model on a filtered test subset.
9. Analyze incorrect predictions.

For EfficientNetB0, the model converts inputs back to [0, 255]
before its built-in preprocessing.

## Models and Validation Results

| Model | Accuracy | Loss | Precision | Recall |
|---|---:|---:|---:|---:|
| One convolution layer | 97.25% | 0.1423 | 98.40% | 97.88% |
| Three convolution layers | 97.05% | 0.1083 | 98.27% | 97.75% |
| Two convolution blocks with augmentation | 74.07% | 0.5773 | 74.07% | 100.00% |
| Frozen EfficientNetB0 | 96.35% | 0.1012 | 98.51% | 96.55% |

These results describe the recorded runs, not guaranteed outcomes
for every rerun. Random initialization can affect training.

The augmented model predicted every validation image as pneumonia
at the default threshold in the recorded final run. Augmentation
did not guarantee an improvement.

The one-convolution-layer CNN was selected before test evaluation
because it had the highest validation accuracy and recall.
The decision threshold was fixed at 0.5.

## Test Filtering

There were 170 inferred filename groups shared between development
and the original test set, affecting 320 test images.

Excluding these images left 304 images. Exact-file hashing found
no matches with development files among those 304 images.
Removing repeated files within the remaining subset left:
- NORMAL: 231 images
- PNEUMONIA: 70 images
- Total: 301 images

Filename groups are not verified patient identities. This filtering
does not establish a fully patient-independent evaluation.

## Final Filtered-Test Results

| Metric | Result |
|---|---:|
| Accuracy | 50.50% |
| Pneumonia precision | 31.96% |
| Pneumonia recall | 100.00% |
| Pneumonia F1-score | 48.44% |

### Confusion Matrix

| Actual / Predicted | NORMAL | PNEUMONIA |
|---|---:|---:|
| NORMAL | 82 | 149 |
| PNEUMONIA | 0 | 70 |

Although all 70 pneumonia images were detected, 149 normal images
were incorrectly classified as pneumonia. Some incorrect
predictions had very high pneumonia scores.

The gap between validation and filtered-test performance shows
poor generalization to this subset. Checks confirmed the selected
model, label order, input range, and decision threshold.
The underlying cause remains unresolved.

## Limitations and Lessons

- High validation accuracy does not guarantee good test performance.
- Precision, recall, F1-score, and confusion matrices reveal
  weaknesses hidden by accuracy alone.
- Filename grouping does not verify patient identity.
- File hashes do not detect all re-encoded or modified duplicates.
- Duplicate checking was performed between development and the
  candidate test subset, and within that test subset; it was not
  a complete near-duplicate audit of training versus validation.
- Filtering changed the test distribution, so these results are
  not directly comparable with the original 624-image benchmark.
- Only 70 pneumonia images remained in the final subset.
- Model scores should not be interpreted as clinically calibrated
  probabilities.

This is an educational project, not a validated diagnostic system.

## Running the Notebook

1. Create a Kaggle notebook and attach the linked dataset.
2. Import `chest-x-ray.ipynb`.
3. Check the dataset path near the beginning of the notebook.
4. Enable internet access to download ImageNet weights.
5. Select a GPU if available and run the cells in order.

Training results may differ between runs.

## Repository Contents

- `chest-x-ray.ipynb`: complete experiment and saved outputs
- `README.md`: project overview, methodology, and results
- `.gitignore`: Python ignore rules
- `LICENSE`: MIT license for repository code

The dataset retains its own licensing terms.

## Author

Zaid Rasras  
Computer Engineering student, Birzeit University
