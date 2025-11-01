This project is my first Convolutional Neural Network (CNN) project.
I used the NIH Chest X-ray Dataset. (https://nihcc.app.box.com/v/ChestXray-NIHCC)
Since this was my first CNN project, I reduced the number of images and merged the original 14 disease categories into 3 (Healthy, Normal disease, Rare disease) to make it manageable.
Even with this reduction, I applied professional techniques to make the project realistic, including:
Proper training, validation, and test splits,
ImageDataGenerator for rescaling and augmentation,
Callbacks(EarlyStopping, ReduceLROnPlateau, ModelCheckpoint),
Balancing the data inside the CNN using class weights,
Building a DenseNet121-based CNN for classification


Results:
Test Accuracy: ~50%
Test Loss: ~1.0
Given the reductions in both dataset size and category simplification, the accuracy is lower than ideal.


Future Work:
I plan to return to this project, use the full dataset with all 14 categories, improve preprocessing and model performance, and aim for higher accuracy.
