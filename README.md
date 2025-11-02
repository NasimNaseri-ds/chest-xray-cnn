# CNN Project — NIH Chest X-ray Dataset

This project is my first **Convolutional Neural Network (CNN)** project. I used the [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC).  

Since this was my first CNN experiment, I reduced the dataset to 22,120 images and merged the original 14 disease categories into 3 classes — **Healthy**, **Normal disease**, and **Rare disease** — to make the project more manageable.  

Even with this reduction, I applied professional deep learning techniques to keep the project realistic and technically solid.  

---

# Data Preparation

- Filtered the dataset to keep only images that actually exist in the local folder (after reduction).  
- Checked for duplicate entries (based on `Image Index`) and missing labels to ensure data consistency.  
- Kept only the necessary columns — `Image Index` and `Finding Labels` — for clarity and efficiency.  
- Grouped the 14 disease categories into 3 main classes:  
  - **Healthy:** images labeled as No Finding  
  - **Normal disease:** common disease categories  
  - **Rare disease:** diseases with fewer than 500 samples  
- Split the dataset into training (70%), validation (15%), and test (15%) sets using stratified sampling to preserve class balance.  
- Created image generators with `ImageDataGenerator`:  
  - Applied rescaling (`1./255`) and moderate augmentation (shear and zoom) for the training set.  
  - Used only rescaling for validation and test sets.  
  - Defined target image size as **224×224 pixels**, with RGB color mode and categorical class mode for multi-class classification.  

---

# Model Building

- Set up essential callbacks for stable and efficient training:  
  - `EarlyStopping`  
  - `ReduceLROnPlateau`  
  - `ModelCheckpoint`  
- Used **DenseNet121** (pre-trained on ImageNet) as the base model and added custom layers on top:  
  - `GlobalAveragePooling2D` to reduce spatial dimensions  
  - `Dense(128, relu)` and `Dropout(0.3)` for regularization  
  - Final `Dense(3, softmax)` layer for the three output categories  
- Handled data imbalance by computing class weights using `compute_class_weight()` from scikit-learn.  
- Trained the model for up to 50 epochs using the prepared `ImageDataGenerator` flows.  
- Evaluated the final saved model (`chestxray.keras`) on the test set.  

---

# Results

- **Test Accuracy:** around 50%  
- **Test Loss:** approximately 1.0  

> Given the reduced dataset size and simplified categories, the accuracy is lower than ideal.  
> The main goal of this project was to practice building, training, and evaluating a full CNN pipeline with professional methods.  

---

# Future Work

I plan to return to this project and:  

- Use the full NIH dataset with all 14 categories  
- Improve preprocessing and model performance  
- Experiment with fine-tuning the DenseNet121 base model  
- Aim for higher accuracy and better generalization  
