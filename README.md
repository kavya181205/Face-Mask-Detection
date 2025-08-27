
# 😷 Face Mask Detection using CNN

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red.svg)

---

## 📌 Overview

This project implements a **Convolutional Neural Network (CNN)** to automatically detect whether a person is wearing a mask or not.
The model is trained on two categories:

* **With Mask (label = 1)**
* **Without Mask (label = 0)**

It achieves reliable accuracy and can be extended for **real-time mask detection**.

---

## 🛠️ Features

* Image preprocessing (resizing, scaling, reshaping)
* CNN model with convolution, pooling, dense, and dropout layers
* Training with **Adam optimizer** and **Sparse Categorical Cross-Entropy loss**
* Evaluation on validation dataset
* Prediction on custom input images

---

## 📂 Project Structure

```
├── data/  
│   ├── with_mask/  
│   └── without_mask/  
├── mask_detection.ipynb   # Jupyter Notebook with code and training  
├── requirements.txt       # Dependencies  
├── README.md              # Project documentation  
```

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
pip install -r requirements.txt
```

---

## ▶️ Usage

1. **Run the Jupyter Notebook**

   ```bash
   jupyter notebook mask_detection.ipynb
   ```

2. **Train the Model**
   Inside the notebook, run the training cell:

   ```python
   history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
   ```

3. **Make Predictions**
   Provide an image path when prompted:

   ```python
   Path of the image to be predicted: test_image.jpg
   ```

   Example Output:

   ```
   The person in the image is wearing a mask
   ```

   or

   ```
   The person in the image is not wearing a mask
   ```

---

## 📊 Model Architecture

* **Conv2D + ReLU**
* **MaxPooling2D**
* **Flatten**
* **Dense layers** with Dropout
* **Output Layer** (2 classes: mask / no mask)

---

## 📈 Results

* Achieved good accuracy with training on \~7,500 images.
* Can classify unseen images correctly into **Mask** or **No Mask**.

---

👨‍💻 Author

Developed by **Kavya Patel** ✨

## 📝 Conclusion

This project demonstrates a practical deep learning application using CNNs for **face mask detection**, which can be extended to:

* Real-time video surveillance
* IoT/Embedded systems
* Public safety monitoring



---

