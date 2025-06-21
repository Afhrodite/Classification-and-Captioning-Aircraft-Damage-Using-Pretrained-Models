import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN optimizations in TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress most TensorFlow messages except actual errors

import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class DatasetManager:
    """
    DatasetManager: loads dataset, prepares generators
    """
    def __init__(self, dataset_path='aircraft_damage_dataset', seed=42, batch_size=32, n_epochs=10):
        self.seed = seed
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.img_rows, self.img_cols = 224, 224 # Set the target image size to 224x224
        self.input_shape = (self.img_rows, self.img_cols, 3) # With 3 channels(RGB)

        self.dataset_path = dataset_path
        self.train_dir = os.path.join(self.dataset_path, 'train')
        self.valid_dir = os.path.join(self.dataset_path, 'valid')
        self.test_dir = os.path.join(self.dataset_path, 'test')

        self.train_generator = None
        self.valid_generator = None
        self.test_generator = None

    def prepare_data_generators(self):
        """
        Initializez the image genetrators that feed images into the neural network during training and evaluation
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255, # Normalize pixel values from [0,255] to [0,1]
            rotation_range=20, # Randomly rotate the images by 20 degrees
            zoom_range=0.15, # Random zoom
            width_shift_range=0.2, 
            height_shift_range=0.2,
            horizontal_flip=True, # Random horizontal flip
            fill_mode="nearest" # Fill empty pixels after shift/rotate by copying nearest ones
        )
        val_test_datagen = ImageDataGenerator(rescale=1./255) # Ensures consistent and realistic evaluation

        # Load training images from the train/ folder
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_rows, self.img_cols), # Resize all to 224x224
            batch_size=self.batch_size,
            class_mode='categorical', # One-hot encoded labels
            seed=self.seed,
            shuffle=True # Shuffle data for training
        )
        # Load valid images from the valid/ folder
        self.valid_generator = val_test_datagen.flow_from_directory(
            self.valid_dir,
            target_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            class_mode='categorical', # One-hot encoded labels
            seed=self.seed,
            shuffle=False # No shuffling
        )
        # Load test images from the test/ folder
        self.test_generator = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )


class ModelBuilder:
    """
    ModelBuilder: builds, compiles, trains the VGG16-based model
    """
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape # Expected shape (224, 224, 3)
        self.num_classes = num_classes # Number of output classes
        self.model = None # Final model (placeholder)

    def build_model(self):
        # Load the VGG16 model pretrained on ImageNet, without the top classification layer
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape) 
        # Freeze all concolutional layers
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x) # Converts feature maps into a single vector
        x = Dense(512, activation='relu')(x) # Fully connected layer with 512 units and ReLU activation
        x = Dropout(0.3)(x) # Reduce overfitting (30%)
        x = Dense(512, activation='relu')(x) # Fully connected layer with 512 units and ReLU activation
        x = Dropout(0.3)(x) # Reduce overfitting (30%)
        output = Dense(self.num_classes, activation='softmax')(x) # Output layer with softmax activation

        self.model = Model(inputs=base_model.input, outputs=output) # Full model: base + custom head
        return self.model

    def compile_model(self):
        self.model.compile(
            optimizer=Adam(), # Optimizer
            loss='categorical_crossentropy',
            metrics=['accuracy'] # Track accuracy
        )

    # Trains the model using the provided data generators
    def train(self, train_generator, valid_generator, epochs):
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=valid_generator
        )
        return history


class Plotter:
    """
    Plotter: helper functions for plotting training curves and images
    """
    @staticmethod
    def plot_loss(history):
        """
        Helps to detect underfitting, overfitting, or convergence issues
        """
        plt.figure()
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_accuracy(history):
        """
        Helps to detect if the model is improving, plateauing, or overfitting.
        """
        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_image_with_title(image, true_label, predicted_label, class_names):
        """
        Visually inspecting misclassifications and confirming that the model “sees” the image meaningfully.
        """
        plt.figure(figsize=(6,6))
        plt.imshow(image)
        true_label_name = class_names[true_label]
        pred_label_name = class_names[predicted_label]
        plt.title(f"True: {true_label_name}\nPredicted: {pred_label_name}")
        plt.axis('off')
        plt.show()


class Evaluator:
    """
    Evaluator: evaluate model and visualize predictions
    """
    def __init__(self, model, test_generator):
        self.model = model # The trained VGG16-based classifier
        self.test_generator = test_generator # Image generator for the test set

    def evaluate(self):
        """
        Final model evaluation on unseen test data
        """
        # Ensures complete batches are processed
        steps = self.test_generator.samples // self.test_generator.batch_size
        # Calculates and prints final loss and accuracy
        loss, accuracy = self.model.evaluate(self.test_generator, steps=steps)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

    def predict_and_show(self, index=0):
        """
        Inspect one prediction at a time
        """
        test_images, test_labels = next(self.test_generator) # Gets a batch of test images
        predictions = self.model.predict(test_images) # Predicts their class probabilities
        # Converts softmax scores to class indices
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        class_indices = self.test_generator.class_indices
        # Maps integer indices to readable class labels
        class_names = {v: k for k, v in class_indices.items()}

        # Visualizes the index-th image from the batch with its true and predicted class labels
        Plotter.plot_image_with_title(
            test_images[index],
            true_classes[index],
            predicted_classes[index],
            class_names
        )


class BLIPCaptioner:
    """
    BLIPCaptioner: generates captions and summaries using BLIP model
    """
    def __init__(self):
        print("Loading BLIP model and processor (this may take some time)...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base") # Handles tokenization and image preprocessing
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base") # Generates image-conditioned text

    def generate_text(self, image_path, task):
        """
        Generates a description or summary of an image
        """
        # Loads an image from disk using its path
        image = Image.open(image_path.numpy().decode('utf-8')).convert('RGB')

        # Chooses a prompt based on the task - caption or summary
        if task.numpy().decode('utf-8') == 'caption':
            prompt = "This is a picture of"
        else:
            prompt = "This is a detailed photo showing"

        # Runs the model and decodes the result
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        output = self.model.generate(**inputs)
        result = self.processor.decode(output[0], skip_special_tokens=True)
        return result

    def tf_wrapper(self, image_path, task):
        """
        Convert to TensorFlow-compatible function
        """
        return tf.py_function(self.generate_text, [image_path, task], tf.string)


def main():
    # Fix random seeds for reproducibility
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # Initialize DatasetManager and prepare data generators
    dataset_manager = DatasetManager(seed=seed_value, batch_size=32, n_epochs=10)
    dataset_manager.prepare_data_generators()

    # Build and compile the model
    num_classes = len(dataset_manager.train_generator.class_indices)
    model_builder = ModelBuilder(dataset_manager.input_shape, num_classes)
    model_builder.build_model()
    model_builder.compile_model()

    # Train the model
    history = model_builder.train(dataset_manager.train_generator, dataset_manager.valid_generator, dataset_manager.n_epochs)

    # Plot training curves
    Plotter.plot_loss(history)
    Plotter.plot_accuracy(history)

    # Evaluate on test data
    evaluator = Evaluator(model_builder.model, dataset_manager.test_generator)
    evaluator.evaluate()

    # Show a prediction example
    evaluator.predict_and_show(index=0)

    # BLIP captioning example on one test image
    blip_captioner = BLIPCaptioner()

    # We can pick any test image path here, example: picks the first test image
    example_image_path = tf.constant(dataset_manager.test_generator.filepaths[0])
    caption = blip_captioner.tf_wrapper(example_image_path, tf.constant("caption"))
    print("Caption:", caption.numpy().decode('utf-8'))

    summary = blip_captioner.tf_wrapper(example_image_path, tf.constant("summary"))
    print("Summary:", summary.numpy().decode('utf-8'))


if __name__ == "__main__":
    main()
