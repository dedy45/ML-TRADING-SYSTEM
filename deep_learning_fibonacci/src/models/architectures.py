"""
Deep Learning Model Architectures for Fibonacci Trading
TensorFlow/Keras implementations optimized for financial time series
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from typing import Tuple, Dict, List, Optional
import yaml
import logging

class FibonacciLSTMModel:
    """
    LSTM Neural Network for Fibonacci level sequence analysis.
    Designed to capture temporal dependencies in trading signals.
    """
    
    def __init__(self, config: Dict):
        self.config = config['models']['lstm']
        self.training_config = config['training']
        self.logger = logging.getLogger(__name__)
        self.model = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: (sequence_length, num_features)
            
        Returns:
            Compiled Keras model
        """
        self.logger.info(f"Building LSTM model with input shape: {input_shape}")
        
        # Input layer
        inputs = keras.Input(shape=input_shape, name='fibonacci_sequence')
        
        # LSTM layers with residual connections
        x = layers.LSTM(
            units=self.config['layers'][0]['lstm_units'],
            return_sequences=self.config['layers'][0]['return_sequences'],
            dropout=self.config['dropout'],
            recurrent_dropout=self.config['dropout'],
            name='lstm_1'
        )(inputs)
        
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        
        x = layers.LSTM(
            units=self.config['layers'][1]['lstm_units'],
            return_sequences=self.config['layers'][1]['return_sequences'],
            dropout=self.config['dropout'],
            recurrent_dropout=self.config['dropout'],
            name='lstm_2'
        )(x)
        
        x = layers.BatchNormalization(name='batch_norm_2')(x)
        
        # Dense layers with attention mechanism
        x = layers.Dense(
            self.config['layers'][2]['dense_units'],
            activation='relu',
            name='dense_1'
        )(x)
        x = layers.Dropout(self.config['dropout'], name='dropout_1')(x)
        
        x = layers.Dense(
            self.config['layers'][3]['dense_units'],
            activation='relu',
            name='dense_2'
        )(x)
        x = layers.Dropout(self.config['dropout'], name='dropout_2')(x)
        
        # Multi-output: signal strength, direction, confidence
        signal_strength = layers.Dense(1, activation='sigmoid', name='signal_strength')(x)
        direction = layers.Dense(1, activation='tanh', name='direction')(x)
        confidence = layers.Dense(1, activation='sigmoid', name='confidence')(x)
        
        # Main binary classification output
        main_output = layers.Dense(1, activation='sigmoid', name='main_output')(x)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs={
                'main_output': main_output,
                'signal_strength': signal_strength,
                'direction': direction,
                'confidence': confidence
            },
            name=self.config['name']
        )
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.training_config['learning_rate']),
            loss={
                'main_output': 'binary_crossentropy',
                'signal_strength': 'mse',
                'direction': 'mse',
                'confidence': 'mse'
            },
            loss_weights={
                'main_output': 1.0,
                'signal_strength': 0.3,
                'direction': 0.2,
                'confidence': 0.2
            },
            metrics={
                'main_output': ['accuracy', 'precision', 'recall'],
                'signal_strength': ['mae'],
                'direction': ['mae'],
                'confidence': ['mae']
            }
        )
        
        self.model = model
        self.logger.info(f"LSTM model built with {model.count_params()} parameters")
        return model
    
    def get_callbacks(self, model_name: str = "fibonacci_lstm") -> List:
        """Get training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_main_output_loss',
                patience=self.training_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_main_output_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'models/saved_models/{model_name}_best.h5',
                monitor='val_main_output_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        return callbacks

class FibonacciCNNModel:
    """
    CNN model for chart pattern recognition at Fibonacci levels.
    Processes candlestick patterns as 2D images.
    """
    
    def __init__(self, config: Dict):
        self.config = config['models']['cnn']
        self.training_config = config['training']
        self.logger = logging.getLogger(__name__)
        self.model = None
        
    def build_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """
        Build CNN model architecture.
        
        Args:
            input_shape: (height, width, channels) - e.g., (32, 32, 4) for OHLC
            
        Returns:
            Compiled Keras model
        """
        self.logger.info(f"Building CNN model with input shape: {input_shape}")
        
        # Input layer
        inputs = keras.Input(shape=input_shape, name='chart_pattern')
        
        # Convolutional blocks with residual connections
        x = layers.Conv2D(
            filters=self.config['layers'][0]['conv2d_filters'],
            kernel_size=self.config['layers'][0]['kernel_size'],
            activation='relu',
            padding='same',
            name='conv_1'
        )(inputs)
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool_1')(x)
        
        x = layers.Conv2D(
            filters=self.config['layers'][1]['conv2d_filters'],
            kernel_size=self.config['layers'][1]['kernel_size'],
            activation='relu',
            padding='same',
            name='conv_2'
        )(x)
        x = layers.BatchNormalization(name='batch_norm_2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool_2')(x)
        
        x = layers.Conv2D(
            filters=self.config['layers'][2]['conv2d_filters'],
            kernel_size=self.config['layers'][2]['kernel_size'],
            activation='relu',
            padding='same',
            name='conv_3'
        )(x)
        x = layers.BatchNormalization(name='batch_norm_3')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool_3')(x)
        
        # Global Average Pooling instead of Flatten to reduce parameters
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Dense layers
        x = layers.Dense(
            self.config['layers'][3]['dense_units'],
            activation='relu',
            name='dense_1'
        )(x)
        x = layers.Dropout(self.config['dropout'], name='dropout_1')(x)
        
        x = layers.Dense(
            self.config['layers'][4]['dense_units'],
            activation='relu',
            name='dense_2'
        )(x)
        x = layers.Dropout(self.config['dropout'], name='dropout_2')(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='pattern_prediction')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=self.config['name'])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.training_config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        self.logger.info(f"CNN model built with {model.count_params()} parameters")
        return model
    
    def get_callbacks(self, model_name: str = "fibonacci_cnn") -> List:
        """Get training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.training_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'models/saved_models/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        return callbacks

class FibonacciEnsembleModel:
    """
    Ensemble model combining LSTM and CNN predictions.
    Uses learned weights to optimally combine model outputs.
    """
    
    def __init__(self, config: Dict):
        self.config = config['models']['ensemble']
        self.training_config = config['training']
        self.logger = logging.getLogger(__name__)
        self.lstm_model = None
        self.cnn_model = None
        self.ensemble_model = None
        
    def build_ensemble(self, lstm_model: Model, cnn_model: Model, 
                      sequence_input_shape: Tuple, image_input_shape: Tuple) -> Model:
        """
        Build ensemble model combining LSTM and CNN.
        
        Args:
            lstm_model: Trained LSTM model
            cnn_model: Trained CNN model
            sequence_input_shape: LSTM input shape
            image_input_shape: CNN input shape
            
        Returns:
            Ensemble model
        """
        self.logger.info("Building ensemble model...")
        
        # Freeze pre-trained models
        lstm_model.trainable = False
        cnn_model.trainable = False
        
        # Input layers
        sequence_input = keras.Input(shape=sequence_input_shape, name='sequence_input')
        image_input = keras.Input(shape=image_input_shape, name='image_input')
        
        # Get predictions from both models
        lstm_pred = lstm_model(sequence_input)
        if isinstance(lstm_pred, dict):
            lstm_pred = lstm_pred['main_output']
        
        cnn_pred = cnn_model(image_input)
        
        # Learned ensemble weights
        lstm_weight = layers.Dense(1, activation='sigmoid', name='lstm_weight')(lstm_pred)
        cnn_weight = layers.Dense(1, activation='sigmoid', name='cnn_weight')(cnn_pred)
        
        # Normalize weights
        total_weight = lstm_weight + cnn_weight + 1e-8  # Avoid division by zero
        lstm_weight_norm = lstm_weight / total_weight
        cnn_weight_norm = cnn_weight / total_weight
        
        # Weighted combination
        ensemble_pred = layers.Add(name='ensemble_prediction')([
            layers.Multiply()([lstm_pred, lstm_weight_norm]),
            layers.Multiply()([cnn_pred, cnn_weight_norm])
        ])
        
        # Final prediction layer
        final_pred = layers.Dense(1, activation='sigmoid', name='final_prediction')(ensemble_pred)
        
        # Create ensemble model
        ensemble_model = Model(
            inputs=[sequence_input, image_input],
            outputs=final_pred,
            name=self.config['name']
        )
        
        # Compile
        ensemble_model.compile(
            optimizer=Adam(learning_rate=self.training_config['learning_rate'] * 0.1),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.ensemble_model = ensemble_model
        self.logger.info(f"Ensemble model built with {ensemble_model.count_params()} parameters")
        return ensemble_model
    
    def predict_with_confidence(self, sequence_data: np.ndarray, 
                              image_data: np.ndarray) -> Dict:
        """
        Make predictions with confidence scores.
        
        Returns:
            Dictionary with prediction, confidence, and individual model scores
        """
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not built yet")
        
        # Get ensemble prediction
        ensemble_pred = self.ensemble_model.predict([sequence_data, image_data])
        
        # Get individual model predictions for analysis
        lstm_pred = self.lstm_model.predict(sequence_data)
        if isinstance(lstm_pred, dict):
            lstm_pred = lstm_pred['main_output']
        
        cnn_pred = self.cnn_model.predict(image_data)
        
        # Calculate confidence based on agreement between models
        agreement = 1 - np.abs(lstm_pred - cnn_pred)
        
        return {
            'prediction': ensemble_pred,
            'confidence': agreement,
            'lstm_prediction': lstm_pred,
            'cnn_prediction': cnn_pred,
            'high_confidence_mask': agreement > self.config['confidence_threshold']
        }

class ModelTrainer:
    """
    Unified trainer for all model types with MLflow integration.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def train_lstm(self, model: FibonacciLSTMModel, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.callbacks.History:
        """Train LSTM model."""
        self.logger.info("Training LSTM model...")
        
        # Prepare multi-output targets for LSTM
        y_train_dict = {
            'main_output': y_train,
            'signal_strength': y_train,  # Simplified - use same target
            'direction': (y_train - 0.5) * 2,  # Convert to [-1, 1]
            'confidence': y_train
        }
        
        y_val_dict = {
            'main_output': y_val,
            'signal_strength': y_val,
            'direction': (y_val - 0.5) * 2,
            'confidence': y_val
        }
        
        history = model.model.fit(
            X_train, y_train_dict,
            batch_size=self.config['training']['batch_size'],
            epochs=self.config['training']['epochs'],
            validation_data=(X_val, y_val_dict),
            callbacks=model.get_callbacks(),
            verbose=1
        )
        
        return history
    
    def train_cnn(self, model: FibonacciCNNModel, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.callbacks.History:
        """Train CNN model."""
        self.logger.info("Training CNN model...")
        
        history = model.model.fit(
            X_train, y_train,
            batch_size=self.config['training']['batch_size'],
            epochs=self.config['training']['epochs'],
            validation_data=(X_val, y_val),
            callbacks=model.get_callbacks(),
            verbose=1
        )
        
        return history

def create_models(config_path: str = "configs/config.yaml"):
    """Factory function to create all models."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    lstm_model = FibonacciLSTMModel(config)
    cnn_model = FibonacciCNNModel(config)
    ensemble_model = FibonacciEnsembleModel(config)
    
    return lstm_model, cnn_model, ensemble_model
