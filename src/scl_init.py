import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2


def spatial_consistency_loss(score_map, heatmap, transformation):
    """
    Computes the Spatial Consistency Loss (SCL).
    Args:
        score_map (tf.Tensor): Current spatial score map (F^t_n), shape [batch_size, G, G, num_classes].
        heatmap (tf.Tensor): Running average heatmap (H^{t-1}_n), shape [batch_size, W, W, num_classes].
        transformation (tf.Tensor): Augmentation transformation applied to the input (e.g., crop, flip).
    Returns:
        tf.Tensor: The SCL value.
    """
    # Apply the transformation (e.g., cropping, flipping) to the heatmap
    transformed_heatmap = tf.image.resize(heatmap, score_map.shape[1:3], method='bilinear')
    # Compute the L1 distance
    scl = tf.reduce_mean(tf.abs(score_map - transformed_heatmap))
    return scl


def custom_loss_function(alpha=1.0, beta=1.0):
    """
    Combines Spatial Consistency Loss and Categorical Cross-Entropy Loss.
    Args:
        alpha (float): Weight for the CCE loss.
        beta (float): Weight for the SCL loss.
    Returns:
        function: A callable loss function to use during model training.
    """

    def loss_function(y_true, y_pred, score_map, heatmap, transformation):
        """
        Args:
            y_true (tf.Tensor): Ground-truth class labels, shape [batch_size, num_classes].
            y_pred (tf.Tensor): Predicted class probabilities, shape [batch_size, num_classes].
            score_map (tf.Tensor): Current spatial score map (F^t_n).
            heatmap (tf.Tensor): Running average heatmap (H^{t-1}_n).
            transformation (tf.Tensor): Transformation applied to input (e.g., crop, flip).
        Returns:
            tf.Tensor: The combined loss value.
        """
        # Compute categorical cross-entropy loss
        cce_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)

        # Compute spatial consistency loss
        scl_loss = spatial_consistency_loss(score_map, heatmap, transformation)

        # Combine the two losses
        total_loss = alpha * cce_loss + beta * scl_loss
        return total_loss

    return loss_function


@tf.function
def train_step(model, optimizer, images, labels, heatmaps, transformation, alpha, beta):
    with tf.GradientTape() as tape:
        # Forward pass: predict class probabilities and score maps
        score_map, class_preds = model(images, training=True)

        # Compute the combined loss
        loss_fn = custom_loss_function(alpha, beta)
        loss = loss_fn(labels, class_preds, score_map, heatmaps, transformation)

    # Backward pass: compute gradients and update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def apply_augmentations(images):
    # Apply random crop, flip, etc.
    return images  # Return transformations applied to images


def get_image_id(image):
    # Logic to uniquely identify images (e.g., from filenames or dataset keys)
    return hash(image.numpy().tostring())  # Example placeholder


def get_initial_heatmap(model, image):
    # Use model's spatial map size to create an empty heatmap
    spatial_shape = (28, 28, num_classes)  # Example size
    return tf.zeros(spatial_shape)


def update_heatmap(current_heatmap, model, image, transformation, momentum=0.8):
    # Predict spatial scores for the image and apply transformation
    spatial_scores = model.predict(image)[0]  # Example prediction
    spatial_scores = tf.image.resize(spatial_scores, current_heatmap.shape[:2])

    # Update running average
    updated_heatmap = momentum * current_heatmap + (1 - momentum) * spatial_scores
    return updated_heatmap


# Example training loop
def train_model(model, dataset, epochs, optimizer, alpha, beta):
    """
    Train the model using the custom train_step function.

    Args:
        model (tf.keras.Model): The neural network model.
        dataset (tf.data.Dataset): Dataset containing input images and labels.
        epochs (int): Number of training epochs.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer for training.
        alpha (float): Weight for categorical cross-entropy loss.
        beta (float): Weight for spatial consistency loss.
    """
    # Initialize heatmaps as a dictionary for each image in the dataset
    heatmaps = {}  # Dictionary to store running-average heatmaps
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0

        for step, (images, labels) in enumerate(tqdm(dataset)):
            batch_size = tf.shape(images)[0]

            # Apply transformations to the input images (e.g., cropping, flipping)
            transformations = apply_augmentations(images)  # Define your augmentation function

            # Initialize or update running-average heatmaps
            if epoch == 0:
                # Initialize heatmaps for each image (zeros except for annotated classes)
                for i in range(batch_size):
                    img_id = get_image_id(images[i])  # Replace with logic to track images
                    if img_id not in heatmaps:
                        heatmaps[img_id] = tf.zeros_like(get_initial_heatmap(model, images[i]))

            # Extract heatmaps for the current batch
            batch_heatmaps = [heatmaps[get_image_id(images[i])] for i in range(batch_size)]

            # Perform a training step
            loss = train_step(
                model=model,
                optimizer=optimizer,
                images=images,
                labels=labels,
                heatmaps=batch_heatmaps,
                transformation=transformations,
                alpha=alpha,
                beta=beta,
            )
            epoch_loss += loss.numpy()

            # Update the heatmaps using the exponential moving average
            for i in range(batch_size):
                img_id = get_image_id(images[i])
                heatmaps[img_id] = update_heatmap(heatmaps[img_id], model, images[i], transformations[i])

        print(f"Epoch {epoch + 1}: Loss = {epoch_loss / (step + 1)}")


def create_model_with_spatial_scores(base_model, num_classes):
    """
    Modifies a given model to output spatial score maps for Spatial Consistency Loss.

    Args:
        base_model (tf.keras.Model): Backbone model without the top dense layers.
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Modified model that outputs spatial score maps.
    """
    # Extract the feature maps from the base model
    feature_maps = base_model.output  # Shape: [batch_size, H, W, C]

    # Add a 1x1 convolutional layer to produce spatial score maps
    spatial_scores = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid', name='spatial_scores')(feature_maps)

    # Optionally add global average pooling for class-level predictions
    class_scores = tf.keras.layers.GlobalAveragePooling2D(name='class_scores')(spatial_scores)

    # Create the model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=[spatial_scores, class_scores])
    return model


# Example Usage
# Load a backbone model without the top dense layers
base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')

# Modify the model for 10 classes
num_classes = 10
model = create_model_with_spatial_scores(base_model, num_classes)

# Print model summary
model.summary()


img_path = '../imgs/moto_gp.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
print(img.shape)
img_array = np.expand_dims(img, axis=0).astype('float32') / 255.0

spatial_scores, class_scores = model.predict([img_array])[0]

plt.imshow(spatial_scores)
plt.show()
print(class_scores)


