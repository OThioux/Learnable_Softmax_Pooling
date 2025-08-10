from warnings import filterwarnings

import cv2
import keras as k
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score


class ClampConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)


class LearnablePooling(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(LearnablePooling, self).__init__()
        # Use self.add_weight() to define a trainable weight
        self.alpha = self.add_weight(
            name="pooling_weight",
            shape=(),  # Scalar weight
            initializer=tf.keras.initializers.Constant(0.5),  # Starting value for alpha
            trainable=True,  # Ensure this is trainable
            # constraint=ClampConstraint(min_value=0.0, max_value=1.0),
        )

    def call(self, spatial_scores):
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(spatial_scores)
        max_pool = tf.keras.layers.GlobalMaxPooling2D()(spatial_scores)
        class_scores = self.alpha * avg_pool + (1 - self.alpha) * max_pool  # Weighted sum
        # class_scores = max_pool
        return class_scores


class GeneralizedMeanPooling(tf.keras.layers.Layer):
    def __init__(self, initial_p=3.0):
        super(GeneralizedMeanPooling, self).__init__()
        self.p = self.add_weight(
            name="p",
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_p),
            trainable=True
        )

    def call(self, inputs, training=True):
        # if not training:
        #     return tf.reduce_max(inputs, axis=[1, 2])
        inputs = tf.maximum(inputs, 1e-6)  # Avoid numerical instability
        pooled = tf.reduce_mean(tf.pow(inputs, self.p), axis=[1, 2])
        return tf.pow(pooled, 1.0 / self.p)


def make_LearnableSoftmaxPooling(use_max=True, start_value=1.0, max_value=2.5):
    class LearnableSoftmaxPooling(tf.keras.layers.Layer):
        def __init__(self):
            super(LearnableSoftmaxPooling, self).__init__()
            self.beta = self.add_weight(
                name="softmax_pooling/beta",
                shape=(),
                initializer=tf.keras.initializers.Constant(start_value),
                trainable=True,
                constraint=ClampConstraint(min_value=-1500, max_value=max_value),
            )

        def call(self, inputs, training=True):
            # if not use_max:
            #     return tf.reduce_mean(inputs, axis=[1, 2])
            if training or not use_max:
                # If beta is inf then this is just the max
                exp_values = tf.exp(self.beta * inputs)
                weights = exp_values / tf.reduce_sum(exp_values, axis=[1, 2], keepdims=True)  # Sums to 1.0
                weighted_avg = tf.reduce_sum(weights * inputs, axis=[1, 2])
                return weighted_avg
            else:
                return tf.reduce_max(inputs, axis=[1, 2])

    return LearnableSoftmaxPooling


def make_LearnableSoftmaxPoolingDouble(use_max=True):
    class LearnableSoftmaxPoolingDouble(tf.keras.layers.Layer):
        def __init__(self):
            super(LearnableSoftmaxPoolingDouble, self).__init__()
            self.beta = self.add_weight(
                name="softmax_pooling/beta",
                shape=(),
                initializer=tf.keras.initializers.Constant(0.001),
                trainable=True
            )
            # self.alpha = self.add_weight(
            #     name="alpha",
            #     shape=(),
            #     initializer=tf.keras.initializers.Constant(0.5),
            #     trainable=True
            # )

        def call(self, inputs, training=True):
            if not use_max:
                return tf.reduce_mean(inputs, axis=[1, 2])
            if training:
                # inputs = inputs - self.alpha
                exp_values = tf.exp(self.beta * inputs)
                weights = exp_values / tf.reduce_sum(exp_values, axis=[1, 2], keepdims=True)
                weighted_avg = tf.reduce_sum(weights * inputs, axis=[1, 2])
                return weighted_avg
            else:
                return tf.reduce_max(inputs, axis=[1, 2])

    return LearnableSoftmaxPoolingDouble


def make_ClassLearnableSoftmaxPooling(use_max=True):
    class ClassLearnableSoftmaxPooling(tf.keras.layers.Layer):
        def __init__(self):
            super(ClassLearnableSoftmaxPooling, self).__init__()
            self.beta = None  # Initialize beta as None

        def build(self, input_shape):
            # The number of channels is the last dimension of the input shape
            num_channels = input_shape[-1]
            self.beta = self.add_weight(
                name="beta",
                shape=(num_channels,),  # Shape is now (num_channels,)
                initializer=tf.keras.initializers.Constant(0.01),
                trainable=True,
                constraint=ClampConstraint(min_value=0.0, max_value=10.0),
            )
            super().build(input_shape)

        def call(self, inputs, training=True):
            if training or not use_max:
                # Expand beta to match the input dimensions for broadcasting
                beta_expanded = tf.expand_dims(tf.expand_dims(self.beta, axis=0), axis=0)  # Shape: (1, 1, num_channels)

                # Multiply each channel by its corresponding beta
                scaled_inputs = beta_expanded * inputs

                exp_values = tf.exp(scaled_inputs)
                weights = exp_values / tf.reduce_sum(exp_values, axis=[1, 2],
                                                     keepdims=True)  # Sums to 1.0 for each channel
                weighted_avg = tf.reduce_sum(weights * inputs, axis=[1, 2])
                return weighted_avg
            else:
                return tf.reduce_max(inputs, axis=[1, 2])

    return ClassLearnableSoftmaxPooling


# pooling_weight = tf.Variable(0.5, trainable=True, name="pooling_weight")
def create_model_with_spatial_scores(base_model, num_classes, use_max, pool_method="softmax", start_value=1.0,
                                     beta_cap_value=2.5):
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
    # spatial_scores = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid', name='spatial_scores')(feature_maps)
    spatial_scores = tf.keras.layers.Conv2D(
        num_classes, (1, 1), activation=None, name='spatial_scores'
    )(feature_maps)
    spatial_scores = tf.nn.sigmoid(spatial_scores)

    # split_feature_maps = tf.split(spatial_scores, num_or_size_splits=num_classes, axis=-1)

    # dense_outputs = []
    # for kernel_out in split_feature_maps:
    #     flattened = tf.keras.layers.Flatten()(kernel_out)
    #     dense_outputs.append(tf.keras.layers.Dense(units=1, activation="sigmoid")(flattened))
    #
    # class_scores = tf.keras.layers.Concatenate(name="class_scores")(dense_outputs)

    # class_scores = tf.keras.layers.GlobalAveragePooling2D(name='class_scores')(spatial_scores)
    # avg_pool = tf.keras.layers.GlobalAveragePooling2D()(spatial_scores)
    # max_pool = tf.keras.layers.GlobalMaxPooling2D()(spatial_scores)
    # alpha = pooling_weight
    # class_scores = alpha * avg_pool + (1 - alpha) * max_pool
    # class_scores = LearnablePooling(num_classes)(spatial_scores)
    if pool_method == "softmax":
        class_scores = (make_LearnableSoftmaxPooling(use_max, start_value=start_value, max_value=beta_cap_value))()(
            spatial_scores)
    elif pool_method == "class_softmax":
        class_scores = (make_ClassLearnableSoftmaxPooling(use_max))()(spatial_scores)
    elif pool_method == "GeM":
        class_scores = GeneralizedMeanPooling()(spatial_scores)
    elif pool_method == "mean":
        class_scores = tf.keras.layers.GlobalAveragePooling2D()(spatial_scores)
    elif pool_method == "max":
        class_scores = tf.keras.layers.GlobalMaxPooling2D()(spatial_scores)
    else:
        raise Exception("Class scores method not valid!")

    # Create the model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=[spatial_scores, class_scores])
    return model


# Suppress warnings that might arise from roc_auc_score/average_precision_score
# when a class has only one label (all 0s or all 1s), which can happen in subsets
# or for rare classes. These cases might result in NaN for that specific class's AUC/AP.
filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')


def calculate_metrics_new(data, cutoff=0.5):
    """
    Calculates various classification metrics for each epoch, including
    accuracy, precision, recall, F1-score, Macro-averaged AUROC, and Macro-averaged mAP.

    Args:
        data (np.ndarray): A NumPy array of shape (epoch, test_sample, class, 2),
                           where the last dimension contains [predicted_probability, real_value].
        cutoff (float): The threshold used to binarize predicted probabilities for
                        metrics like accuracy, precision, recall, and F1-score.

    Returns:
        dict: A dictionary containing lists of values for each metric per epoch.
              Keys include "accuracy", "precision", "recall", "f1", "auroc", and "map".
              Note: AUROC and mAP are calculated using raw probabilities and are
              macro-averaged across classes.
    """
    num_epochs = data.shape[0]
    num_samples = data.shape[1]  # Number of test samples
    num_classes = data.shape[2]

    accuracy_per_epoch = []
    precision_per_epoch = []
    recall_per_epoch = []
    f1_per_epoch = []
    auroc_per_epoch = []  # New list for Macro-averaged AUROC
    map_per_epoch = []  # New list for Macro-averaged mAP

    for epoch in range(num_epochs):
        epoch_data = data[epoch]

        # Extract raw predicted probabilities and ground truth labels
        # Shape: (num_samples, num_classes)
        predictions_raw = epoch_data[:, :, 0]
        ground_truth = epoch_data[:, :, 1]

        # --- Metrics requiring a cutoff (Accuracy, Precision, Recall, F1) ---
        # Binarize predictions using the cutoff
        predictions_binarized = (predictions_raw > cutoff).astype(float)

        # Accuracy calculation (Micro-accuracy / Overall accuracy on flattened labels)
        # Compares all individual (sample, class) predictions against ground truth
        correct_predictions = (predictions_binarized == ground_truth)
        accuracy = np.sum(correct_predictions) / correct_predictions.size
        accuracy_per_epoch.append(accuracy)

        # Precision, Recall, F1 calculation (Macro-averaged)
        # Calculated per class, then averaged.
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []

        for cls in range(num_classes):
            pred_cls = predictions_binarized[:, cls]
            true_cls = ground_truth[:, cls]

            true_positive = np.sum((pred_cls == 1) & (true_cls == 1))
            false_positive = np.sum((pred_cls == 1) & (true_cls == 0))
            false_negative = np.sum((pred_cls == 0) & (true_cls == 1))

            # Handle division by zero for precision/recall/f1
            precision = true_positive / (true_positive + false_positive) if (
                                                                                        true_positive + false_positive) > 0 else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)

        # Average across all classes for this epoch (Macro-averaging)
        precision_per_epoch.append(np.mean(precision_per_class))
        recall_per_epoch.append(np.mean(recall_per_class))
        f1_per_epoch.append(np.mean(f1_per_class))

        # --- Metrics independent of a specific cutoff (AUROC, mAP) ---
        # These use the raw predicted probabilities (predictions_raw)

        # Calculate Macro-averaged AUROC
        # sklearn's roc_auc_score can compute this directly for multi-label.
        # It handles cases where a class might have no positive or negative samples
        # by excluding them from the average, or returning NaN if all are excluded.
        try:
            # 'macro' average computes AUC for each label, then takes the unweighted mean.
            # This is suitable for your multi-label, 20-class scenario.
            auroc = roc_auc_score(ground_truth, predictions_raw, average='macro')
        except ValueError as e:
            # Catch specific errors (e.g., all samples for a class are negative/positive)
            # which can make AUC undefined. Assign NaN to indicate it couldn't be computed.
            # A UserWarning is often issued by sklearn in these cases, which we suppressed.
            auroc = np.nan
            # print(f"Warning: Could not compute AUROC for epoch {epoch}: {e}")
        auroc_per_epoch.append(auroc)

        # Calculate Macro-averaged mAP (Mean Average Precision)
        # average_precision_score computes AP for each label, then takes the unweighted mean.
        try:
            # 'macro' average computes AP for each label, then takes the unweighted mean.
            map_score = average_precision_score(ground_truth, predictions_raw, average='macro')
        except ValueError as e:
            map_score = np.nan
            # print(f"Warning: Could not compute mAP for epoch {epoch}: {e}")
        map_per_epoch.append(map_score)

    return {
        "accuracy": accuracy_per_epoch,
        "precision": precision_per_epoch,
        "recall": recall_per_epoch,
        "f1": f1_per_epoch,
        "AUROC": auroc_per_epoch,  # New metric
        "mAP": map_per_epoch  # New metric
    }


def calculate_metrics(data, cutoff=0.5):
    """
    Calculates accuracy and precision for each epoch.

    Args:
        data (np.ndarray): A NumPy array of shape (epoch, test_sample, class, 2),
                           where the last dimension contains [predicted value, real value].

    Returns:
        dict: A dictionary with keys "accuracy" and "precision", each containing a list
              of values for each epoch.
    """
    num_epochs = data.shape[0]
    num_classes = data.shape[2]

    accuracy_per_epoch = []
    precision_per_epoch = []
    recall_per_epoch = []
    f1_per_epoch = []

    for epoch in range(num_epochs):
        epoch_data = data[epoch]

        # Flatten across test samples
        predictions = epoch_data[:, :, 0]
        predictions = np.asarray((predictions > cutoff), dtype=float)
        ground_truth = epoch_data[:, :, 1]

        # Accuracy calculation
        correct_predictions = (predictions == ground_truth)
        accuracy = np.sum(correct_predictions) / correct_predictions.size
        accuracy_per_epoch.append(accuracy)

        # Precision calculation per class
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        for cls in range(num_classes):
            pred_cls = predictions[:, cls]
            true_cls = ground_truth[:, cls]

            true_positive = np.sum((pred_cls == 1) & (true_cls == 1))
            false_positive = np.sum((pred_cls == 1) & (true_cls == 0))
            false_negative = np.sum((pred_cls == 0) & (true_cls == 1))

            if true_positive + false_positive > 0:
                precision = true_positive / (true_positive + false_positive)
            else:
                precision = 0.0  # Avoid division by zero

            if true_positive + false_negative > 0:
                recall = true_positive / (true_positive + false_negative)
            else:
                recall = 0.0

            if recall + precision > 0:
                f1 = (recall * precision) / (recall + precision)
            else:
                f1 = 0.0

            recall_per_class.append(recall)
            precision_per_class.append(precision)
            f1_per_class.append(f1)

        # Average precision across all classes for this epoch
        precision_per_epoch.append(np.mean(precision_per_class))
        recall_per_epoch.append(np.mean(recall_per_class))
        f1_per_epoch.append(np.mean(f1_per_class))

    return {
        "accuracy": accuracy_per_epoch,
        "precision": precision_per_epoch,
        "recall": recall_per_epoch,
        "f1": f1_per_epoch
    }


def to_categorical(x):
    no_classes = 20
    return np.asarray(k.utils.to_categorical(x, no_classes + 1), dtype="?")


def to_ordinal(x):
    og_shape = x.shape
    x_new = np.zeros(x.shape[0] * x.shape[1])
    x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
    for idx, vals in enumerate(x):
        x_new[idx] = np.argmax(vals)
    return np.reshape(x_new, og_shape[:2])


def make_augement_image_and_mask(transform, model_input_shape, seed=None):
    def augment_image_and_mask(image, mask):
        img_smallest_dim = np.argmin(image.shape[:2])
        ratio = model_input_shape[img_smallest_dim] / image.shape[img_smallest_dim]
        new_size = (np.ceil(ratio * np.asarray(image.shape[:2]))).astype(int)
        if len(new_size) != 2:
            print(new_size)
            print("REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n\n\n")
        try:
            image = tf.image.resize(image, size=new_size, method=tf.image.ResizeMethod.BILINEAR)
            mask = tf.image.resize(mask[..., tf.newaxis], size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        except ValueError as e:
            print(new_size)
            print(e)
            image = tf.image.resize(image, size=model_input_shape[:2], method=tf.image.ResizeMethod.BILINEAR)
            mask = tf.image.resize(mask[..., tf.newaxis], size=model_input_shape[:2],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        mask = tf.squeeze(mask, axis=-1)  # Remove the added channel dimension
        image, mask = np.asarray(image.numpy()), np.asarray(mask.numpy())
        if seed:
            transform.set_random_seed(seed)
        transformed = transform(image=np.asarray(image), mask=to_categorical(np.asarray(mask)).astype(int))
        image, mask = transformed["image"], transformed["mask"]
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1.0
        return tf.convert_to_tensor(image, dtype=tf.float32), tf.convert_to_tensor(mask, dtype=tf.uint8)

    return augment_image_and_mask


def make_resize_to_output_tf(model_input_shape, model_feature_map_shape):
    def resize_to_output_tf(seg_masks, out_shape):
        """
        Applies max pooling to downsample a batch of one-hot encoded segmentation masks.

        Args:
            seg_masks (tf.Tensor): Batch of one-hot encoded segmentation masks
                                   with shape (batch_size, height, width, num_classes).
            out_shape (tuple): Target spatial dimensions as (target_height, target_width).

        Returns:
            tf.Tensor: Downsampled segmentation maps with shape (batch_size, target_height, target_width, num_classes).
        """

        seg_masks = tf.cast(seg_masks, dtype=tf.float32)

        # if seg_masks.shape[-1] != 20:
        #     seg_masks = seg_masks[:,:,:,1:]
        input_shape = model_input_shape
        input_height = input_shape[0]
        input_width = input_shape[1]

        # Compute pooling sizes and strides
        pool_size_height = input_height // model_feature_map_shape[0]
        pool_size_width = input_width // model_feature_map_shape[1]

        # Use eager execution values
        if tf.executing_eagerly():
            pool_size = [1, int(pool_size_height), int(pool_size_width), 1]
            strides = [1, int(pool_size_height), int(pool_size_width), 1]
        else:
            # Use tf.get_static_value to resolve tensors to static values
            # pool_size_height = tf.get_static_value(pool_size_height)
            # pool_size_width = tf.get_static_value(pool_size_width)
            pool_size = [1, pool_size_height, pool_size_width, 1]
            strides = [1, pool_size_height, pool_size_width, 1]

        # Max pooling
        downsampled = tf.nn.max_pool(
            seg_masks,
            ksize=pool_size,
            strides=strides,
            padding="VALID"
        )
        return downsampled

    return resize_to_output_tf


def seg_from_box(box, img_shape, no_classes=21):
    seg = np.zeros(img_shape[:-1])
    classes = np.zeros(no_classes)
    for b in box:
        seg[b[1]:b[3], b[0]:b[2]] = b[4]
        classes[b[4] - 1] = 1
    return seg, classes
