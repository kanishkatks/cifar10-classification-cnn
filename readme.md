# CIFAR10 Image Classification with Convolutional Neural Networks

## Project Goals
This project aims to build and train a Convolutional Neural Network (CNN) to classify images from the CIFAR10 dataset. The dataset consists of 60,000 32x32 color images in 10 different classes: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

The main objectives are:

1. **Implement a CNN** architecture for classifying the 10 categories in CIFAR10.
2. **Improve model performance** using data augmentation techniques.


## Methods Used

### 1. Data Loading and Preprocessing:
- The CIFAR10 dataset is loaded using Keras.
- Images are preprocessed by normalizing pixel intensities to the range [0, 1].
- Labels are one-hot encoded for use with the model.

### 2. Model Architecture and Training:
- A custom CNN model is built using Keras layers, with focus on keeping the number of parameters under 150,000.
- The model is compiled with the Adam optimizer and categorical cross-entropy loss function.
- Training is done on a smaller subset of the dataset for initial experimentation.
- Early stopping is used to prevent overfitting.

### 3. Data Augmentation:
- Keras' `ImageDataGenerator` is used to perform data augmentation techniques like rotation, shifting, flipping, and zooming.
- Augmentation is applied on-the-fly during training to expand the training dataset and improve model generalization.
- The validation set is kept separate to avoid data leakage.


## Results

- The initial model trained on the smaller dataset achieves a certain accuracy on the test set.
- After implementing data augmentation, the model performance further improves, showing better generalization capabilities.
- The training and validation accuracy/loss curves are plotted to visually assess the model's performance.

## Future Improvements

- Experiment with different CNN architectures like ResNet or EfficientNet to potentially further improve accuracy.
- Fine-tune hyperparameters such as learning rate and batch size to find optimal settings.
- Explore more advanced data augmentation techniques or pre-trained models.

## Acknowledgments

- The CIFAR10 dataset is provided by the Canadian Institute For Advanced Research.
- Keras and TensorFlow libraries are used for building and training the model.



