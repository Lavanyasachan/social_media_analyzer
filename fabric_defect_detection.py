"""
Fabric Defect Detection using ResNet CNN Model

This module implements a ResNet50-based deep learning model for detecting fabric defects
using transfer learning. It provides a complete pipeline for downloading data, training,
and evaluating a CNN classifier.

Main Class:
    FabricDefectDetector: Complete fabric defect detection system

Key Features:
    - ResNet50 transfer learning architecture
    - Automatic Kaggle dataset download
    - Data augmentation pipeline
    - Model training with callbacks
    - Comprehensive evaluation (accuracy, confusion matrix)
    - Visualization of results

Basic Usage:
    >>> from fabric_defect_detection import FabricDefectDetector
    >>> detector = FabricDefectDetector(img_size=(224, 224), batch_size=32)
    >>> dataset_path = detector.download_dataset()
    >>> train_gen, val_gen = detector.prepare_data_generators(dataset_path)
    >>> detector.build_resnet_model(num_classes=len(detector.class_names))
    >>> detector.train_model(train_gen, val_gen, epochs=50)
    >>> results = detector.evaluate_model(val_gen)
    >>> detector.plot_confusion_matrix(results['confusion_matrix'])

Command Line Usage:
    $ python fabric_defect_detection.py

Requirements:
    - TensorFlow >= 2.10.0
    - Keras >= 2.10.0
    - NumPy, Pandas, Matplotlib, Seaborn
    - scikit-learn
    - kagglehub
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import kagglehub


class FabricDefectDetector:
    """
    A class for building and training a ResNet-based fabric defect detection model
    """
    
    def __init__(self, img_size=(224, 224), batch_size=32):
        """
        Initialize the Fabric Defect Detector
        
        Args:
            img_size (tuple): Size to resize images to (height, width)
            batch_size (int): Batch size for training
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None
        
    def download_dataset(self):
        """
        Download the fabric defects dataset from Kaggle
        
        Returns:
            str: Path to the downloaded dataset
        """
        print("Downloading fabric defects dataset from Kaggle...")
        path = kagglehub.dataset_download("nexuswho/fabric-defects-dataset")
        print(f"Path to dataset files: {path}")
        return path
    
    def prepare_data_generators(self, dataset_path, validation_split=0.2):
        """
        Prepare data generators for training and validation
        
        Args:
            dataset_path (str): Path to the dataset directory
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            tuple: (train_generator, validation_generator)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create training generator
        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Create validation generator
        validation_generator = val_datagen.flow_from_directory(
            dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {validation_generator.samples}")
        
        return train_generator, validation_generator
    
    def build_resnet_model(self, num_classes):
        """
        Build a ResNet-based model for fabric defect detection
        
        Args:
            num_classes (int): Number of defect classes
            
        Returns:
            keras.Model: Compiled ResNet model
        """
        # Load pre-trained ResNet50 without top layers
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size[0], self.img_size[1], 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build the model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        print("ResNet model built successfully!")
        print(f"\nModel Summary:")
        model.summary()
        
        return model
    
    def train_model(self, train_generator, validation_generator, epochs=50):
        """
        Train the ResNet model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs (int): Number of training epochs
            
        Returns:
            History: Training history
        """
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'best_fabric_defect_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("\nStarting model training...")
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def fine_tune_model(self, train_generator, validation_generator, epochs=20):
        """
        Fine-tune the model by unfreezing some layers of ResNet
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs (int): Number of fine-tuning epochs
        """
        # Unfreeze the base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Freeze all layers except the last 20
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompile with a lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print("\nFine-tuning the model...")
        print(f"Trainable layers: {sum([1 for layer in self.model.layers if layer.trainable])}")
        
        # Continue training
        history_fine = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            verbose=1
        )
        
        return history_fine
    
    def evaluate_model(self, validation_generator):
        """
        Evaluate the model and generate metrics
        
        Args:
            validation_generator: Validation data generator
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print("\nEvaluating model...")
        
        # Get predictions
        validation_generator.reset()
        y_pred_probs = self.model.predict(validation_generator, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = validation_generator.classes
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        print(f"\n{'='*50}")
        print(f"MODEL EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def plot_training_history(self, save_path='training_history.png'):
        """
        Plot training history
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix
        
        Args:
            cm (numpy.ndarray): Confusion matrix
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Fabric Defect Detection', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
        plt.show()
    
    def save_model(self, filepath='fabric_defect_model.h5'):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            print("No model to save!")
            return
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='fabric_defect_model.h5'):
        """
        Load a trained model
        
        Args:
            filepath (str): Path to the model file
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def main():
    """
    Main function to run the fabric defect detection pipeline
    """
    print("="*60)
    print("FABRIC DEFECT DETECTION USING RESNET CNN")
    print("="*60)
    
    # Initialize detector
    detector = FabricDefectDetector(img_size=(224, 224), batch_size=32)
    
    # Download dataset
    dataset_path = detector.download_dataset()
    
    # Prepare data generators
    train_gen, val_gen = detector.prepare_data_generators(dataset_path, validation_split=0.2)
    
    # Get number of classes
    num_classes = len(detector.class_names)
    
    # Build model
    model = detector.build_resnet_model(num_classes)
    
    # Train model
    history = detector.train_model(train_gen, val_gen, epochs=50)
    
    # Plot training history
    detector.plot_training_history()
    
    # Evaluate model
    results = detector.evaluate_model(val_gen)
    
    # Plot confusion matrix
    detector.plot_confusion_matrix(results['confusion_matrix'])
    
    # Save model
    detector.save_model('fabric_defect_resnet_model.h5')
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nFinal Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print("\nModel saved as: fabric_defect_resnet_model.h5")
    print("Training history plot: training_history.png")
    print("Confusion matrix plot: confusion_matrix.png")


if __name__ == "__main__":
    main()
