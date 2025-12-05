import tensorflow as tf
import numpy as np
from PIL import Image
from weights import download_weights
from model import download_model
import cv2 as cv
import gradio as gr

def main():
  def load_model():
    download_model()
    model=tf.keras.models.load_model("model/model.h5")
    model.compile(optimizer =tf.keras.optimizers.Adam(learning_rate=0.00001,decay=0.0001),metrics=["accuracy"],
                  loss= tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1))
    download_weights()
    model.load_weights("weights/modeldense1.h5")
    return model

  model=load_model()
  
  def preprocess(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    return cv.filter2D(image, -1, kernel)

  
  image = gr.Image(width=224, height=224)

  # label=gr.Label(num_top_classes=8)
  label = gr.Label(num_top_classes=2)

  
  class_name=['Benign with Density=1','Malignant with Density=1','Benign with Density=2','Malignant with Density=2','Benign with Density=3','Malignant with Density=3','Benign with Density=4','Malignant with Density=4']
  
  # def predict_img(img):
  #   # Ensure PIL -> numpy
  #   if isinstance(img, Image.Image):
  #       img = np.array(img)

  #   # Resize BEFORE sharpening
  #   img = cv.resize(img, (224, 224))

  #   # Sharpen
  #   kernel = np.array([[0, -1, 0],
  #                      [-1, 5, -1],
  #                      [0, -1, 0]])
  #   img = cv.filter2D(img, -1, kernel)

  #   # Normalize
  #   img = img / 255.0

  #   # Batch dimension
  #   img = img.reshape(1, 224, 224, 3)

  #   pred = model.predict(img)[0]
  #   return {class_name[i]: float(pred[i]) for i in range(8)}

  def predict_img(img):
      # Ensure PIL -> numpy
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Resize BEFORE sharpening
    img = cv.resize(img, (224, 224))

    # Sharpen
    kernel = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
    img = cv.filter2D(img, -1, kernel)

    # Normalize
    img = img / 255.0

    # Batch dimension
    img = img.reshape(1, 224, 224, 3)

    pred = model.predict(img)[0]   # shape (8,)

    # 8 CLASS ORDER YOU DEFINED:
    # 0: Benign D1
    # 1: Malignant D1
    # 2: Benign D2
    # 3: Malignant D2
    # 4: Benign D3
    # 5: Malignant D3
    # 6: Benign D4
    # 7: Malignant D4

    benign_sum = float(pred[0] + pred[2] + pred[4] + pred[6])
    malignant_sum = float(pred[1] + pred[3] + pred[5] + pred[7])

    result = {
        "Benign": benign_sum,
        "Malignant": malignant_sum
    }

    return result

  gr.Interface(fn=predict_img, inputs=image, outputs=label).launch(debug=True, share=True)
  
if __name__=='__main__':
    main()
    
