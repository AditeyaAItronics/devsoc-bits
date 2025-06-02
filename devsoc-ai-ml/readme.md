## Projects (Any 1 of the below)

### 📊 **Ensemble Learning**

### 🧠 **Obesity Risk Prediction Using Ensemble Learning**

**🎯 Objective:**

Predict an individual’s risk of obesity using machine learning. You’ll work with a structured dataset containing health, lifestyle, and demographic features. Your goal is to classify individuals into different obesity risk categories by analysing patterns and building effective predictive models.

---

### 📦 Dataset

A link to the dataset is provided at the bottom of this document. The data includes various features relevant to obesity prediction, such as eating habits, physical activity, and personal health indicators.

---

### 🔍 What You'll Do

You are required to experiment with and explore **ensemble learning techniques**—methods that combine multiple models to improve accuracy and robustness over individual models.

### ✅ Baseline

- Begin by building and evaluating at least **one traditional ML model** (e.g., Decision Tree, Logistic Regression, KNN).
- Use this as a benchmark for comparing ensemble performance.

### 🚀 Ensemble Techniques to Explore

---

### **1. Bagging (Bootstrap Aggregating)**

- Trains several models independently on different random subsets of the data.
- Final prediction: **majority vote (classification)** or **average (regression)**.
- 📌 *Example:* Random Forest

---

### **2. Boosting**

- Models are trained **sequentially**, each focusing on the previous model’s errors.
- Final prediction: **weighted combination** of all models.
- 📌 *Popular Algorithms:* AdaBoost, Gradient Boosting, XGBoost, LightGBM

---

### **3. Stacking (Stacked Generalization)**

- Combines multiple base models using a **meta-learner** trained on their outputs.
- Base models learn from the original data; the meta-learner learns from their predictions.

---

### **4. Blending**

- A simpler form of stacking using a **hold-out validation set** instead of cross-validation.

---

### **5. Voting & Weighted Voting**

- Combine predictions from multiple models:
  - **Hard Voting:** Majority class wins
  - **Soft Voting:** Average predicted probabilities
  - **Weighted Voting:** Give more influence to better-performing models

---

### **6. Model Diversity**

Improve ensemble performance by using a diverse set of base models:

- Different algorithms
- Different hyperparameters
- Different subsets of features

---

### 📊 Evaluation Criteria

- Split the dataset into **training and testing sets** (e.g., 80/20 split or use cross-validation).
- Use key metrics to evaluate performance:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
  - **Confusion matrix** for multi-class evaluation
- Clearly present your results using **tables and/or plots**.
- Compare ensemble methods against your **baseline model**.

---

### 🔗 Dataset : <https://drive.google.com/file/d/1k1YVdYwc-kVtUDjfTi18PnqGZtaeHrOF/view>

### 🖼️ Neural Style Transfer: Intro to Deep Learning & CNNs

## 🧠 Neural Style Transfer using VGG-19

In this task, you’ll implement **Neural Style Transfer (NST)** using a **pre-trained VGG-19** model. This will help you deepen your understanding of convolutional neural networks (CNNs), feature extraction, and how deep learning can be used for image generation.

---

### ✅ Objectives

- Build a Neural Style Transfer pipeline using **VGG-19**.
- Use only the **first convolutional layer** from **each of the five convolutional blocks** in VGG-19.
  - These layers strike a balance between general texture features and higher-level abstraction.
  - Deeper layers become too specialized for object recognition and are less effective for capturing style.

---

### ⚙️ System Requirements

VGG-19 is a **computationally heavy** model. If your system struggles to run it:

- Use **Google Collab** to access free GPU resources.
- This will allow faster computation and smoother experimentation.

---

### 📚 Learning Goals

This task is more than just implementation. Take time to understand the theory and architecture:

- How **CNNs** extract and represent features at different depths
- What the **VGG-19** model architecture looks like
- How **style** and **content** are represented in NST
- Loss functions used in NST:
  - **Content Loss**
  - **Style Loss** (via Gram Matrices)
  - **Total Variation Loss** (optional, for smoothing)

---

### 📄 Reference Paper

We’ve linked the paper on **CNN-based image style transformation** below.

It uses VGG-19 and provides insight into the theory behind NST.

> 🔗 ‣
>

Please read it carefully—it will help you understand what’s going on inside the model and how different layers contribute to the stylization process.

---

### 💬 Interview Tip

You may be asked to explain:

- How the VGG-19 architecture works
- Why specific layers are used
- How the model separates and recombines content and style

    Be prepared to discuss the intuition behind your implementation choices.

---

### 📤 Deliverables

- A working NST implementation (preferably in a Jupyter/Colab notebook)
- Stylized output image(s)
- Brief documentation explaining:
  - The role of each key component
  - Why certain layers were used
  - What you observed or experimented with

---

### 🎨 Bonus

Experiment with different images, layer weights, or learning rates to see how they affect the output. The more you explore, the more you’ll learn.

### 🧠 Neural Networks from Scratch (MNIST Project)

## 🧠 MNIST Digit Classification – Build a Neural Network from Scratch

This project challenges you to **build a neural network from the ground up**, without relying on any machine learning libraries or frameworks. You'll work with the classic **MNIST dataset**, which contains 60,000 training and 10,000 test grayscale images (28×28 pixels) of handwritten digits (0–9).

---

### 🎯 Objective

Build a model that can accurately classify handwritten digits by implementing all the fundamental components of a neural network manually. This includes:

- Creating custom structures for **neurons** and **layers**
- Writing code for **forward propagation**
- Implementing **backpropagation** to compute gradients and update weights
- Experimenting with different **activation functions** and **loss functions**
- Training using **gradient descent** and observing how various hyperparameters affect performance

---

### 🧱 What You’ll Implement

### ✅ Core Components

- **Neurons & Layers**: Create classes to represent individual neurons and layer connections.
- **Forward Propagation**: Manually compute outputs as data flows through layers using chosen activation functions.
- **Backpropagation**: Calculate gradients for each weight and bias using the chain rule and update parameters.

### ⚙️ Features to Experiment With

- **Activation Functions**:
  - Sigmoid
  - Tanh
  - ReLU
- **Loss Functions**:
  - Mean Squared Error (MSE)
  - Cross-Entropy Loss
- **Optimization**:
  - Stochastic Gradient Descent (SGD)
- **Initialization**:
  - Random weight and bias initialization

---

### 📚 Learning Outcomes

- Understand the **mechanics behind neural networks**
- Learn how **forward and backward propagation** work internally
- Explore the impact of **different loss functions, optimizers, and activations**
- Build a strong foundation for diving deeper into **deep learning and AI**

---

### 🗂 Resources

📁 **Drive Link**: <https://drive.google.com/drive/folders/1DwGuH6ZDl2LENmu4m1cTYY5_V_kHtBVB?usp=sharing>

The folder contains:

- The MNIST dataset (images and labels)
- A reference PDF that explains key terms and concepts

We know these projects might seem **challenging and complex**, especially if it’s your first time working with machine learning from scratch. And that’s completely okay.

**Ideally**, we expect you to **complete the full project**, applying the concepts you’ve learned to build and train your model effectively.

But if you're not able to finish everything, **don’t stress** — just **submit whatever you've done so far**. We’ll be evaluating your **effort, thought process, and learning**, not just the final output.

Focus on **understanding, experimenting, and asking questions**—that’s what truly matters.

### Submission Link : [Gform link](https://forms.gle/T74AdebPLTd7zejb8)
