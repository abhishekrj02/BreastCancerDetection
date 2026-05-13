import io
import base64
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


def build_grad_model(model):
    """
    Build a model that outputs (last_conv_4d_features, final_predictions)
    from the same forward pass, enabling proper Grad-CAM.
    model: Sequential [DenseNet201, BN, Dense(2048), BN, Dense(8)]
    """
    densenet = model.layers[0]

    # Find last layer with 4D output (spatial feature maps) in DenseNet201
    last_conv_layer = None
    for layer in reversed(densenet.layers):
        try:
            shape = layer.output.shape
            if len(shape) == 4:
                last_conv_layer = layer
                break
        except Exception:
            continue

    if last_conv_layer is None:
        return None, None

    # Build sub-model: densenet_input → (4D conv features, pooled 1D features)
    conv_out = last_conv_layer.output
    pooled_out = densenet.output  # 1D after GlobalMaxPool, flows through conv_out

    # Continue pooled_out through remaining Sequential layers (BN → Dense → BN → Dense)
    x = pooled_out
    for layer in model.layers[1:]:
        x = layer(x)

    # final model: densenet_input → (spatial_features, full_predictions)
    grad_model = tf.keras.Model(
        inputs=densenet.input,
        outputs=[conv_out, x]
    )
    return grad_model, last_conv_layer.name


_grad_model_cache = None


def compute_gradcam(model, img_array, class_idx=None):
    """
    Compute Grad-CAM heatmap for the given image.
    Returns: (heatmap_2d_normalized, pred_class_idx) or (None, None) on failure.
    """
    global _grad_model_cache
    try:
        if _grad_model_cache is None:
            _grad_model_cache, _ = build_grad_model(model)
        if _grad_model_cache is None:
            return None, None

        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = _grad_model_cache(img_tensor)
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))
            class_score = predictions[:, class_idx]

        grads = tape.gradient(class_score, conv_outputs)

        if grads is None:
            # Fallback: saliency map (gradient w.r.t. input)
            return _saliency_fallback(model, img_array, class_idx), class_idx

        # Grad-CAM: global-average-pool the gradients → weight the feature maps
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        conv_np = conv_outputs[0].numpy()

        for i in range(pooled_grads.shape[0]):
            conv_np[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_np, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, class_idx

    except Exception as e:
        print(f"[Grad-CAM] Error: {e}")
        try:
            return _saliency_fallback(model, img_array, class_idx or 0), class_idx or 0
        except Exception:
            return None, None


def _saliency_fallback(model, img_array, class_idx):
    """Compute a simple saliency map as fallback."""
    img_var = tf.Variable(tf.cast(img_array, tf.float32))
    with tf.GradientTape() as tape:
        preds = model(img_var)
        loss = preds[0, class_idx]
    grads = tape.gradient(loss, img_var)[0]
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()
    if saliency.max() > 0:
        saliency /= saliency.max()
    return saliency


def overlay_heatmap(heatmap, original_img_bytes, alpha=0.45):
    """
    Overlay Grad-CAM heatmap on the original image.
    Returns: data-URI base64 PNG string.
    """
    img = Image.open(io.BytesIO(original_img_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    img_np = np.array(img)

    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(img_np, 1 - alpha, heatmap_rgb, alpha, 0)

    out = Image.fromarray(superimposed.astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
