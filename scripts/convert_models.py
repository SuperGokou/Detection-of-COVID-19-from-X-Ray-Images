"""
Convert trained Keras models to TensorFlow.js layers model format.

Usage:
    cd Web
    pip install tensorflowjs
    python scripts/convert_models.py

This script:
1. Rebuilds each model architecture exactly as in covid19_run.py
2. Loads trained weights (searches trained_models/, ../models/, ../models_polished/)
3. Converts to TF.js layers model via save_keras_model()
4. Outputs to public/models/{custom-cnn,densenet121,resnet50}/

Layers model format is required for Grad-CAM (needs gradient + layer access).

Weight file naming convention (from get_callbacks in covid19_run.py):
    best_{model_name}.weights.h5
    For transfer learning, Phase 2 weights contain the fine-tuned model:
        best_densenet121_phase2.weights.h5
        best_resnet50_phase2.weights.h5
"""

import os
import sys

# Add project root to path so we can potentially reuse code
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121, ResNet50

# Monkey-patch tensorflowjs version check to work with Keras 3
import tensorflowjs.converters.keras_h5_conversion as _keras_h5
_keras_h5._check_version = lambda h5file: None

# Look for weights in multiple locations
_candidates = [
    os.path.join(PROJECT_ROOT, "trained_models"),       # Web/trained_models/
    os.path.join(PROJECT_ROOT, "..", "models"),          # Covid_Project/models/
    os.path.join(PROJECT_ROOT, "..", "models_polished"), # Covid_Project/models_polished/
]
WEIGHTS_DIR = next((d for d in _candidates if os.path.isdir(d)), _candidates[0])
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "public", "models")

IMAGE_SIZE = 224


def build_custom_cnn(input_shape=(224, 224, 1)):
    """Build a custom CNN for binary classification.
    Exact copy from covid19_run.py lines 635-678."""
    reg = tf.keras.regularizers.l2(1e-4)

    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg,
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ], name='Custom_CNN')

    return model


def build_densenet121(input_shape=(224, 224, 3)):
    """Build DenseNet121 transfer learning model.
    Exact copy from covid19_run.py lines 701-714.
    Returns model in Phase 2 state (last 30 layers unfrozen)."""
    base = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base.input, outputs=outputs, name='DenseNet121')

    # Replicate Phase 2 state: unfreeze last 30 layers of base
    # (covid19_run.py lines 744-746)
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    return model


def build_resnet50(input_shape=(224, 224, 3)):
    """Build ResNet50 transfer learning model.
    Exact copy from covid19_run.py lines 769-782.
    Returns model in Phase 2 state (last 30 layers unfrozen)."""
    base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base.input, outputs=outputs, name='ResNet50')

    # Replicate Phase 2 state: unfreeze last 30 layers of base
    # (covid19_run.py lines 812-814)
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    return model


# Model configurations: (build_fn, weight_filename, output_dir)
MODEL_SPECS = [
    {
        "name": "Custom CNN",
        "build_fn": build_custom_cnn,
        "build_args": {"input_shape": (IMAGE_SIZE, IMAGE_SIZE, 1)},
        "weight_file": "best_custom_cnn.weights.h5",
        "output_dir": "custom-cnn",
    },
    {
        "name": "DenseNet121",
        "build_fn": build_densenet121,
        "build_args": {"input_shape": (IMAGE_SIZE, IMAGE_SIZE, 3)},
        "weight_file": "best_densenet121_phase2.weights.h5",
        "output_dir": "densenet121",
    },
    {
        "name": "ResNet50",
        "build_fn": build_resnet50,
        "build_args": {"input_shape": (IMAGE_SIZE, IMAGE_SIZE, 3)},
        "weight_file": "best_resnet50_phase2.weights.h5",
        "output_dir": "resnet50",
    },
]


def convert_model(spec):
    """Convert a single model to TF.js layers model format."""
    name = spec["name"]
    weight_path = os.path.join(WEIGHTS_DIR, spec["weight_file"])
    output_dir = os.path.join(OUTPUT_BASE, spec["output_dir"])

    print(f"\n{'='*60}")
    print(f"Converting {name}")
    print(f"{'='*60}")

    # Check weight file exists
    if not os.path.exists(weight_path):
        print(f"  [SKIP] Weight file not found: {weight_path}")
        return False

    # Build model
    print(f"  Building model architecture...")
    model = spec["build_fn"](**spec["build_args"])
    model.summary(print_fn=lambda x: print(f"    {x}"))

    # Load weights
    print(f"  Loading weights from: {weight_path}")
    model.load_weights(weight_path)

    # Convert directly to TF.js layers model
    print(f"  Converting to TF.js layers model -> {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    from tensorflowjs.converters.keras_h5_conversion import save_keras_model
    save_keras_model(model, output_dir)

    # Report output size
    total_size = 0
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            print(f"    {f}: {size / 1024:.1f} KB")
    print(f"    Total: {total_size / (1024 * 1024):.1f} MB")

    return True


def main():
    print("TensorFlow version:", tf.__version__)
    print(f"Weights directory: {WEIGHTS_DIR}")
    print(f"Output directory: {OUTPUT_BASE}")

    # List available weight files
    if os.path.exists(WEIGHTS_DIR):
        print(f"\nAvailable weight files in {WEIGHTS_DIR}:")
        for f in os.listdir(WEIGHTS_DIR):
            print(f"  - {f}")
    else:
        print(f"\nWeight directory not found: {WEIGHTS_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    results = []
    for spec in MODEL_SPECS:
        success = convert_model(spec)
        results.append((spec["name"], success))

    # Summary
    print(f"\n{'='*60}")
    print("Conversion Summary")
    print(f"{'='*60}")
    for name, success in results:
        status = "OK" if success else "SKIPPED"
        print(f"  {name}: {status}")

    converted_count = sum(1 for _, s in results if s)
    print(f"\n{converted_count}/{len(results)} models converted successfully.")

    if converted_count == 0:
        print("\nNo models were converted. Make sure weight files exist in:")
        print(f"  {WEIGHTS_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    main()
