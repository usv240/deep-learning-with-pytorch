# PyTorch & Deep Learning

---

## Introduction
PyTorch is a popular deep learning framework used by teams at companies like Tesla, Meta, and OpenAI. It lets you write Python code that runs fast on GPUs, so you can train models on images, text, audio, and more without wrestling with low‑level math. The goal of these notes is to learn by **doing**: write small pieces of code, see what happens, and build up your intuition step by step.

**In a nutshell:** PyTorch makes it easy to write and run deep learning code in Python, especially on GPUs.

---

## What is Machine Learning?
Machine Learning (ML) is a way to teach computers by example. We turn real‑world stuff, like pictures, sentences, or sound, into numbers. Then an algorithm looks for patterns in those numbers so it can make predictions on new data.

A simple way to think about it: humans learn from examples (lots of them), and so do ML models. If we show enough examples of cats and dogs, a model can learn the differences and identify a new image it has never seen before.

**In a nutshell:** ML = turn data into numbers, then learn patterns to predict new cases.

---

## AI vs ML vs Deep Learning
- **Artificial Intelligence (AI):** The broad idea of making computers act intelligently.
- **Machine Learning (ML):** A subset of AI where systems learn from data rather than being hard‑coded.
- **Deep Learning (DL):** A subset of ML that uses **neural networks** with many layers (hence “deep”) to learn complex patterns, great for images, text, and audio.

In practice, people sometimes use “ML” and “DL” interchangeably. What matters is choosing the right tool for your data and problem.

**In a nutshell:** DL is a specialized kind of ML that uses multi‑layer neural networks to learn complex patterns.

---

## Traditional Programming vs Machine Learning
**Traditional programming** is: Inputs + Hand‑written Rules → Output.  
Example: a recipe, ingredients (inputs) + cooking steps (rules) → finished dish (output). The rules are explicit; you wrote them.

**Machine Learning** flips it: Inputs + Desired Outputs → **Algorithm learns the rules**.  
Give the model many examples (images + correct labels), and it figures out what patterns connect inputs to outputs. You don’t write the recognition rules by hand, the model learns them.

**In a nutshell:** Traditional = you write rules. ML = the model learns rules from examples.

---

## Why Use Machine Learning?
Some problems have **too many rules** to write down. Driving a car is a classic example: recognizing signs, predicting other drivers, handling lighting and weather, there are thousands of “if this then that” rules. Instead of writing all of them, we can show examples and let a model discover the patterns.

**In a nutshell:** Use ML when the rulebook would be huge or constantly changing.

---

## When to Use ML or DL
Deep learning shines when:
- The problem is complex and the “rules” are hard to spell out.
- The environment changes and the system needs to adapt over time.
- You have a **lot** of data, especially unstructured data (images, text, audio).

**In a nutshell:** DL is best for complex, evolving problems with plenty of data.

---

## When Not to Use Deep Learning
- **You need clear explanations:** Many DL models are “black boxes.” If you must explain every decision (e.g., for regulations), DL might be a poor fit.
MATRIX = torch.tensor([[7, 8],
                       [9,10]])
print(MATRIX.ndim)   # 2
print(MATRIX.shape)  # torch.Size([2, 2])
**In a nutshell:** Avoid DL when you need transparency, simple rules suffice, errors are unacceptable, or data is scarce.

---
tensor = torch.tensor([[[1,2,3],
                        [4,5,6],
                        [7,8,9]]])
print(tensor.ndim)   # 3
print(tensor.shape)  # torch.Size([1, 3, 3])
**In a nutshell:** Tables → traditional ML. Images/text/audio → deep learning.

random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape)
# torch.Size([224, 224, 3])
**Common traditional ML algorithms (great for structured data):**
- Gradient Boosted Trees (e.g., XGBoost)
- Random Forests
one_to_ten = torch.arange(start=1, end=11, step=1)
ten_zeros = torch.zeros_like(input=one_to_ten)
ten_zeros

**Common deep learning architectures (great for unstructured data):**
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float32)
int_32_tensor   = torch.tensor([3, 6, 9], dtype=torch.int32)
- **RNNs / sequence models:** handle ordered data like time series or text.
- **Transformers:** state‑of‑the‑art for language and increasingly for vision, audio, and multimodal tasks.
tensor_a = torch.tensor([[1, 2],
                         [3, 4]])
tensor_b = torch.tensor([[5, 6],
                         [7, 8]])

print(torch.matmul(tensor_a, tensor_b))
# or equivalently: tensor_a @ tensor_b

**In a nutshell:** “Deep” means many layers that learn increasingly abstract features.
tensor = torch.tensor([1, 2, 3])
print(tensor * tensor)  # tensor([1, 4, 9])

## Neural Networks (Basics)
tensor_a = torch.tensor([[1, 2, 3]])    # shape: [1, 3]
tensor_b = torch.tensor([[1], [2], [3]])# shape: [3, 1]
print(torch.matmul(tensor_a, tensor_b))        # tensor([[14]]) because 1*1 + 2*2 + 3*3 = 14
# Shorthand: tensor_a @ tensor_b
- **Non‑linearities:** Functions like ReLU or sigmoid let networks model curved, complex relationships, not just straight lines.
- **Depth & width:** You can choose how many layers (depth) and how many units per layer (width) to suit the problem.
tensor_a = torch.rand(2, 3)
tensor_b = torch.rand(3, 4)
output = tensor_a @ tensor_b           # output.shape == torch.Size([2, 4])

tensor_c = torch.rand(2, 3)
# tensor_c @ tensor_c -> RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)

## Learning Paradigms
x = torch.arange(1., 10.)         # shape: [9]
x_reshaped = x.reshape(1, 9)      # [1, 9]
x_reshaped.shape
**In a nutshell:** Labeled data teaches the model what’s what.

x = torch.randn(1, 9)
x = x.squeeze()                   # shape: [9]
x = x.unsqueeze(0)                # shape: [1, 9]
x = x.unsqueeze(1)                # shape: [9, 1]
  Example: A folder of cat and dog photos without labels. The model might cluster similar images, even if it doesn’t know the words “cat” or “dog.”

random_image_size_tensor = torch.rand(size=(224, 224, 3))
# Convert to CHW (channels, height, width)
random_image_size_tensor = random_image_size_tensor.permute(2, 0, 1)   # [3, 224, 224]

---

### Transfer Learning
Deep learning models often require lots of data. Instead of training from scratch, we reuse a model already trained on a huge dataset (like ImageNet) and fine‑tune it for our task.  
Example: Start with a model that knows general image features, then tweak it for cat vs dog classification.

**In a nutshell:** Transfer learning = borrow knowledge from a bigger task, then fine‑tune.

---

### Reinforcement Learning
In reinforcement learning (RL), an **agent learns by interacting with an environment** and receiving rewards or penalties.  
Example: Training a dog. If it pees outside → reward. Inside → no reward or penalty. Over time, the dog learns the “policy” that maximizes rewards.

**In a nutshell:** RL = learn by trial and error with rewards and penalties.

---

## Applications of Deep Learning

Deep learning can be applied almost anywhere we can turn things into numbers. Some popular areas:  

- **Recommendation Systems:** Netflix, YouTube, and Spotify suggest content based on your history and preferences.  
- **Machine Translation:** Google Translate now uses deep learning for much better results compared to older rule-based systems.  
- **Speech Recognition:** Assistants like Siri and Alexa convert audio into text commands.  
- **Computer Vision:** Detecting objects in photos/videos (e.g., cars, faces, products).  
- **Natural Language Processing (NLP):** Spam filters, chatbots, and sentiment analysis.

### Task Types
- **Classification:** Predict categories (binary or multiclass), e.g., spam vs not‑spam, cat vs dog.
- **Regression:** Predict continuous values, e.g., bounding‑box coordinates or prices.
- **Sequence‑to‑Sequence (Seq2Seq):** Map input sequences to output sequences, e.g., translation or speech‑to‑text.
- **Object Detection/Segmentation:** Locate and/or delineate objects in images (boxes or pixel‑wise masks).

**In a nutshell:** DL powers many real-world tools, translation, recommendations, vision, speech, NLP.

---

## PyTorch Introduction

PyTorch is one of the most widely used deep learning frameworks.  
- **Easy to use:** It feels like writing normal Python code.  
- **GPU acceleration:** Runs heavy computations on GPUs, making it fast.  
- **Flexibility:** Great for both research and production.  
- **Community:** Huge ecosystem, with pretrained models and active forums.  

Companies like Tesla (Autopilot), Microsoft, Meta, and OpenAI rely heavily on PyTorch.

**In a nutshell:** PyTorch is the go-to framework for deep learning, both in research and industry.

---

## Tensors

A **tensor** is the basic building block in PyTorch. Think of it as a multi‑dimensional array (like a generalization of vectors and matrices).  
- **Input:** Your data (image, text, audio) is turned into tensors.  
- **Model:** Neural networks process tensors through layers and weights.  
- **Output:** Predictions are also tensors, later converted into human‑readable form.

PyTorch makes tensor operations easy and efficient, especially with GPU acceleration.

**In a nutshell:** Tensors = the universal data format deep learning models use.

---

## Course Structure (High-Level)

The learning workflow usually follows these steps:  
1. Get data ready (and convert into tensors).  
2. Pick or build a model.  
3. Choose a loss function + optimizer.  
4. Train the model (fit to data).  
5. Evaluate performance.  
6. Improve the model with experiments.  
7. Save and reload trained models.  
8. Use models on new, real-world data.

**In a nutshell:** Data → Model → Train → Evaluate → Improve → Deploy.

---

## Best Practices for Learning

- **Code along:** The best way to learn ML is to actually run code.  
- **Experiment:** Don’t just copy, tweak numbers, try different layers, see what changes.  
- **Visualize:** Plots and charts help you understand what’s happening inside the model.  
- **Ask questions:** No question is too basic, curiosity is key.  
- **Do exercises:** Applying concepts on your own cements understanding.  
- **Share your work:** Writing or posting about your learning helps memory and showcases your skills.

**In a nutshell:** Learn by coding, experimenting, visualizing, and sharing.

---

## Resources

- **Official PyTorch Docs (pytorch.org):** The ultimate reference.  
- **Forums (discuss.pytorch.org):** Ask questions and see others’ solutions.  
- **GitHub repo (for course):** Contains code, notebooks, and exercises.  
- **Online book (learnpytorch.io):** Organized, searchable notes generated from notebooks.

**In a nutshell:** Use the official docs + community to go deeper when stuck.

---

## PyTorch Setup (Google Colab)

Google Colab is the easiest way to start coding with PyTorch:  
- Runs in the browser, no installation needed.  
- Free GPU support (you can enable it in settings).  
- Preloaded with Python, PyTorch, and data science libraries.  
- Paid upgrade = faster GPUs and more reliability.  

Example: Run `!nvidia-smi` in Colab to check which GPU you got.  

**In a nutshell:** Colab = free cloud lab for PyTorch with GPUs ready to go.

---

## Working with Tensors

Tensors are the backbone of PyTorch. They’re like NumPy arrays but with extra superpowers: they can run on GPUs.  
- **Creating tensors:** You can create them from Python lists, NumPy arrays, or directly with PyTorch functions.  
- **Shapes and dimensions:** A tensor’s shape tells you how many dimensions it has (e.g., scalars, vectors, matrices, or higher‑dimensional tensors).  
- **Common operations:** addition, multiplication, reshaping, slicing, indexing.  

Example:  
- A 1D tensor is like a list `[1, 2, 3]`.  
- A 2D tensor is like a matrix of rows and columns.  
- A 3D tensor could represent color images (height × width × channels).  

**In a nutshell:** Tensors = flexible containers of numbers that run fast on GPUs.

---

## Preprocessing Data

Before feeding data into a model, we need to make sure it’s in the right form:  
1. **Collect raw data** (images, text, audio, etc.).  
2. **Convert it into numbers** → this often means tokenizing text, normalizing image pixel values, or turning categories into one‑hot vectors.  
3. **Wrap it into tensors** so PyTorch can handle it.  
4. **Batch the data** → models train more efficiently in small batches rather than one example at a time.  

**In a nutshell:** Preprocessing = clean raw data, convert into tensors, and batch it for training.

---

## Building Models

In PyTorch, you define models using `nn.Module`.  
- **Layers:** Building blocks like linear layers, convolutional layers, or recurrent layers.  
- **Activation functions:** Non‑linear transformations (ReLU, sigmoid, tanh) that let models learn complex patterns.  
- **Forward pass:** Defines how input flows through the layers to produce an output.  

Think of a model as a function: **input → transformations → output.**

**In a nutshell:** A PyTorch model is a stack of layers defined in a class that extends `nn.Module`.

---

## Training Models

Training is where the learning happens:  
1. **Loss function:** Measures how wrong the model is.  
2. **Optimizer:** Updates the model’s weights to reduce the loss (e.g., SGD, Adam).  
3. **Training loop:**  
   - Forward pass: input → model → prediction.  
   - Compute loss by comparing predictions with actual labels.  
   - Backward pass: calculate gradients using autograd.  
   - Optimizer step: update weights.  

Repeat this process for many epochs until performance improves.  

**In a nutshell:** Training = forward pass → compute loss → backward pass → update weights.

---

## Making Predictions

Once trained, the model can make predictions:  
- Give it new data → it produces outputs (like probabilities or labels).  
- Example: A trained food classifier can look at a new image and predict “sushi” or “pizza.”  

Predictions are often post‑processed (e.g., using softmax to convert raw scores into probabilities).  

**In a nutshell:** Predictions = feed new data through the trained model to get answers.

---

## Evaluating Models

After training, we need to measure how good the model is:  
- **Accuracy:** For classification (percentage of correct predictions).  
- **Precision/Recall/F1:** For imbalanced datasets where accuracy isn’t enough.  
- **Loss curves:** Plot loss over time to see if the model is learning.  
- **Validation data:** Always test on data the model hasn’t seen before.  

**In a nutshell:** Evaluation checks if the model generalizes well to unseen data.

---

## Saving and Loading Models

Training can take hours or days, so saving progress is important.  
- **Save:** `torch.save(model.state_dict(), "model.pth")`  
- **Load:** Create the same model structure and call `load_state_dict`.  

This lets you reuse models later or share them with others.  

**In a nutshell:** Save trained weights so you don’t have to retrain from scratch.

---

## Real-World Usage

After training and evaluation, the model can be deployed:  
- **Applications:** Use the trained model in web apps, mobile apps, or backend services.  
- **Transfer learning:** Start with pretrained models from Torch Hub and fine‑tune on your own data.  
- **Scaling:** Run models on servers, cloud platforms, or even edge devices.  

**In a nutshell:** Trained models move from notebooks to real apps where they make predictions in the wild.

---

## Experimentation Mindset

Deep learning is part science, part art.  
- **Scientist mindset:** Form hypotheses and test them systematically.  
- **Chef mindset:** Add a “pinch of this, dash of that,” and see what happens.  

Don’t be afraid to experiment with:  
- Different numbers of layers.  
- Different activation functions.  
- Changing learning rates or optimizers.  
- Trying out new architectures.  

Often, you learn more by playing than by following strict rules.  

**In a nutshell:** Treat deep learning like cooking and science combined, experiment freely.

---

## Visualization

Models work with lots of numbers. Visualizing these numbers makes them understandable.  
- **Loss curves:** Show if the model is learning or overfitting.  
- **Prediction plots:** Compare predictions vs. true labels.  
- **Feature visualizations:** Look at learned filters in CNNs.  

Tools like Matplotlib, TensorBoard, or other visualization libraries help you “see” what’s happening inside your model.  

**In a nutshell:** Visuals turn abstract numbers into insights.

---

## Debugging & Asking Questions

Every ML journey involves confusion and errors. That’s normal.  
- **Debugging approach:** Write small experiments to test your ideas. Let the code answer your questions.  
- **Ask questions early:** In forums, course discussions, or Google.  
- **Remember:** If you’re confused, others probably are too.  

**In a nutshell:** Debug with small experiments, and never hesitate to ask questions.

---

## Exercises

At the end of each lesson or module, try the exercises.  
- They force you to apply what you just learned.  
- Struggling through them deepens your understanding.  
- They stretch your thinking beyond just copying code.  

**In a nutshell:** Exercises turn passive watching into active learning.

---

## Sharing Your Work

Writing or publishing what you’ve learned has double benefits:  
1. **For you:** Explaining something out loud or in writing makes you remember it better.  
2. **For others:** Someone else struggling might find your notes helpful.  
3. **For employers:** A GitHub repo or blog shows your learning journey and projects.  

Places to share: GitHub, LinkedIn, personal blogs, or Discord communities.  

**In a nutshell:** Sharing multiplies your learning and helps others too.

---

## Mindset Tips

- Avoid overthinking. Learning DL is a journey.  
- Don’t get discouraged if things don’t click right away.  
- Stay curious and explore.  
- Step-by-step progress beats frustration and burnout.  

Think of it like building muscle, small consistent reps are more effective than occasional huge efforts.  

**In a nutshell:** Stay patient, curious, and consistent, learning compounds over time.

---

## Setting Up the Environment

The easiest way to start coding with PyTorch is using **Google Colab** — a free, cloud‑based notebook that lets you write and run Python right in your browser.  
Colab comes pre‑installed with **PyTorch**, **NumPy**, **Pandas**, and **Matplotlib**, so you don’t need to set anything up.

To enable GPU acceleration (for faster training):

```python
# Go to: Runtime → Change runtime type → Hardware accelerator → GPU
```

You can check your GPU details with:
```python
!nvidia-smi
```
This shows which GPU Colab has assigned to you (often Tesla T4 or P100).  

**In a nutshell:** Google Colab gives you a ready‑to‑use PyTorch setup with free GPU support.

---

## What Are Tensors?

In deep learning, **everything revolves around tensors.**  
A tensor is just a way to store numbers in multiple dimensions — like an upgraded version of lists and arrays.

| Concept | Example | Dimensions | Description |
|----------|----------|-------------|--------------|
| Scalar | `torch.tensor(7)` | 0D | A single number |
| Vector | `torch.tensor([1, 2, 3])` | 1D | A list of numbers |
| Matrix | `torch.tensor([[1, 2], [3, 4]])` | 2D | Rows × columns grid |
| Tensor | `torch.tensor([[[1,2],[3,4]]])` | 3D+ | Stack of matrices |

Each tensor has:  
- **`shape`** → size in each dimension  
- **`ndim`** → number of dimensions  
- **`dtype`** → data type (float, int, etc.)

Example – Creating a Scalar:

```python
scalar = torch.tensor(7)
scalar.item()
```
`.item()` converts the one‑element tensor to a normal Python number.

**In a nutshell:** Tensors are multi‑dimensional containers of numbers — the foundation of deep learning.

---

## Creating Vectors and Matrices

### Example: Vector
```python
vector = torch.tensor([3, 6, 9])
print(vector.ndim)   # 1
print(vector.shape)  # torch.Size([3])
```
A 1‑D tensor (vector) stores a simple list of values.

### Example: Matrix
```python
matrix = torch.tensor([[7, 8],
                       [9,10]])
print(matrix.ndim)   # 2
print(matrix.shape)  # torch.Size([2, 2])
```
A 2‑D tensor is like a spreadsheet — rows × columns.

**In a nutshell:** 1D → vector, 2D → matrix; each extra bracket adds a dimension.

---

## Creating a 3‑D Tensor

A 3‑D tensor is a collection of 2‑D matrices stacked together.

```python
tensor3d = torch.tensor([[[1,2,3],
                          [4,5,6],
                          [7,8,9]]])
print(tensor3d.ndim)   # 3
print(tensor3d.shape)  # torch.Size([1, 3, 3])
```
These are used for color images (height × width × channels) or batches of data.

**In a nutshell:** Each new set of `[]` adds a dimension; 3D tensors often represent images or data batches.

---

## Random Tensors — How Learning Starts

Models start from random numbers and learn by adjusting them gradually.  
You can create a random tensor like this:

```python
random_tensor = torch.rand(3, 4)
print(random_tensor)
```
This makes a 3×4 tensor filled with random numbers between 0 and 1.

Why this matters: when you train a neural network, the initial weights are random. As the model learns, those random values get tuned into meaningful patterns.

**In a nutshell:** Every neural network starts with random numbers and learns to adjust them through training.

---

## Image‑Shaped Tensors

Images are stored as numeric arrays of pixel values.  
Most color images are 3‑D tensors with the shape *(height, width, channels)*.

```python
image = torch.rand(224, 224, 3)
print(image.shape)
# torch.Size([224, 224, 3])
```
Here, 224×224 is the resolution and 3 represents the RGB color channels.

**In a nutshell:** Images = 3‑D tensors (height × width × color channels).

---

## Zero and One Tensors

Tensors filled with zeros or ones are useful for initialization or masking.

```python
zeros = torch.zeros(3, 4)
ones  = torch.ones(3, 4)
```
- All elements = 0 or 1.  
- You can multiply another tensor by 0 to “mask out” parts you want to ignore.

**In a nutshell:** `torch.zeros()` and `torch.ones()` create uniform tensors for testing or masking.

---

## Creating Ranges of Numbers

Use `torch.arange(start, end, step)` to generate a sequence of numbers.

```python
torch.arange(0, 10)      # 0 1 2 3 4 5 6 7 8 9
torch.arange(1, 10, 2)   # 1 3 5 7 9
```
This is similar to Python’s `range()` but returns a tensor.

**In a nutshell:** `torch.arange()` creates sequences of numbers inside tensors.

---

## Tensors “Like” Other Tensors

You can quickly create new tensors with the same shape as an existing one.

```python
base = torch.arange(1, 10)
zeros_like = torch.zeros_like(base)
ones_like  = torch.ones_like(base)
```
These new tensors match `base` in shape but have different values.

**In a nutshell:** “_like” functions clone the shape of another tensor without manually typing dimensions.

---

## Tensor Data Types (dtype)

Data types control how precise and memory‑hungry numbers are.

| dtype | Description | Typical Use |
|--------|--------------|-------------|
| `torch.float32` | 32‑bit float (single precision) | Default for most tasks |
| `torch.float16` | 16‑bit float (half precision) | Faster on GPU, less accurate |
| `torch.int32` / `torch.int64` | Integers | Counting or indexing |
| `torch.bool` | True/False values | Masks and conditions |

Example:

```python
float_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float32)
int_tensor   = torch.tensor([3, 6, 9], dtype=torch.int32)
```

**In a nutshell:** `dtype` defines how precise each number is and how much memory it uses.

---

## Common Tensor Errors and Fixes

Typical mistakes in PyTorch:

1. **Mismatched data types** (e.g., float + int).  
2. **Mismatched shapes** (different dimensions).  
3. **Mismatched devices** (one tensor on CPU, another on GPU).

Example – Fixing dtype:

```python
t1 = torch.tensor([1, 2, 3], dtype=torch.float32)
t2 = torch.tensor([4, 5, 6], dtype=torch.float16)

t2 = t2.to(torch.float32)
result = t1 + t2
```
Now both tensors match in dtype, and addition works.

**In a nutshell:** Check `dtype`, `shape`, and `device` first when something breaks.

---

## Devices — CPU vs GPU

PyTorch tensors can live on two devices: **CPU** or **GPU**.

By default, tensors are created on the CPU:
```python
tensor = torch.rand(3, 4)
print(tensor.device)  # cpu
```

If your computer (or Colab) has a GPU, you can move tensors to it:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor = tensor.to(device)
print(tensor.device)
```

Why this matters:
- GPUs can perform thousands of calculations in parallel.
- Moving tensors to GPU speeds up training dramatically.
- But CPU↔GPU transfers take time — keep everything on one device during computation.

**In a nutshell:** CPUs handle general tasks; GPUs crunch numbers in parallel for deep learning.

---

## Tensor Attributes and Inspection

Every tensor has a few important attributes:
- **`dtype`** – data type (e.g., float32, int64)
- **`shape`** – rows × columns × … dimensions
- **`device`** – where it lives (CPU or GPU)
- **`ndim`** – number of dimensions

Example:
```python
t = torch.rand(3, 4)
print(t.dtype)   # torch.float32
print(t.shape)   # torch.Size([3, 4])
print(t.device)  # cpu
print(t.ndim)    # 2
```

Tip: `.shape` and `.size()` do the same thing.

**In a nutshell:** Always check a tensor’s dtype, shape, and device before heavy computations.

---

## Basic Tensor Operations

Tensors support standard math directly — addition, subtraction, multiplication, and division.

### Element‑wise Arithmetic
Each element is processed individually:
```python
x = torch.tensor([1, 2, 3])
print(x + 10)  # tensor([11, 12, 13])
print(x * 2)   # tensor([2, 4, 6])
print(x / 2)   # tensor([0.5, 1.0, 1.5])
```

### Using PyTorch Functions
PyTorch also provides functional forms that behave identically:
```python
print(torch.add(x, 10))
print(torch.mul(x, 2))
print(torch.sub(x, 1))
```

These are useful when chaining multiple operations in model code.

**In a nutshell:** PyTorch math works element‑by‑element — you can use operators (`+`, `*`) or function calls (`torch.add`, `torch.mul`).

---

## Matrix Multiplication Preview

Matrix multiplication (`@` or `torch.matmul`) is the most important operation in neural networks — it’s how features and weights interact.

```python
A = torch.tensor([[1, 2],
                  [3, 4]])
B = torch.tensor([[5, 6],
                  [7, 8]])

print(torch.matmul(A, B))
# or equivalently: A @ B
```
Output:
```
tensor([[19, 22],
        [43, 50]])
```

**How it works:**  
Each row of `A` is multiplied with each column of `B`, and the products are summed — producing a new matrix.

**Visual:**
```
A (2×2)  @  B (2×2)  →  Result (2×2)
```

This operation is the foundation of layers in a neural network — inputs (data) × weights (parameters) = outputs (features).

**In a nutshell:** Matrix multiplication mixes data and learned weights — it’s the math heart of deep learning.

---

## Matrix Multiplication: Element‑Wise vs Matrix (Dot Product)

There are two main kinds of multiplication you’ll use with tensors:

1) Element‑wise multiplication (Hadamard product) – multiply values position‑by‑position.

```python
import torch
x = torch.tensor([1, 2, 3])
print(x * x)  # tensor([1, 4, 9])
```

2) Matrix multiplication (dot product) – combine rows and columns to mix information across features. This is the core operation in neural networks.

```python
A = torch.tensor([[1, 2, 3]])    # shape: [1, 3]
B = torch.tensor([[1], [2], [3]])# shape: [3, 1]
print(torch.matmul(A, B))        # tensor([[14]]) because 1*1 + 2*2 + 3*3 = 14
# Shorthand: A @ B
```

Tip: Use `torch.matmul` (or `@`) for neural network math. It’s highly optimized (vectorized) and much faster than manual Python loops.

In a nutshell: Element‑wise = independent positions; matrix multiplication = mixes features via row‑by‑column dot products.

---

## Matrix Multiplication Rules (Shapes That Work)

Matrix multiplication follows two simple but crucial shape rules:

- Rule 1: Inner dimensions must match.
  - Example: [2 × 3] @ [3 × 4] → works; [2 × 3] @ [2 × 3] → error.
- Rule 2: The result has the shape of the outer dimensions.
  - Example: [2 × 3] @ [3 × 4] → result is [2 × 4].

```python
A = torch.rand(2, 3)
B = torch.rand(3, 4)
C = A @ B           # C.shape == torch.Size([2, 4])

D = torch.rand(2, 3)
# D @ D -> RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)
```

In a nutshell: Check inner dims match; the output keeps the outer dims.

---

## Fixing Shape Errors with Transpose

If shapes don’t line up, transpose one tensor to swap its axes.

```python
A = torch.tensor([[1, 2],
                  [3, 4],
                  [5, 6]])      # shape: [3, 2]
B = torch.tensor([[ 7, 10],
                  [ 8, 11],
                  [ 9, 12]])     # shape: [3, 2]

# Inner dims don’t match: (3,2) @ (3,2) -> error
# Transpose B to (2,3):
BT = B.T                        # same numbers, axes swapped
out = A @ BT                    # (3,2) @ (2,3) -> (3,3)
print(out.shape)                # torch.Size([3, 3])
```

Notes:
- `.T` (or `transpose`) rearranges dimensions; values don’t change, just their layout.
- Common error message: “mat1 and mat2 shapes cannot be multiplied” → fix by transposing or reshaping so inner dims match.

In a nutshell: When shapes don’t multiply, transpose to swap axes and satisfy the inner‑dims rule.

---

## Performance: Use Vectorized Ops (Not Loops)

Matrix multiplication with `@`/`torch.matmul` runs in optimized C/CUDA. Manual Python loops are far slower, and the gap grows with tensor size. Prefer vectorized PyTorch ops for speed and clarity.

In a nutshell: Trust `@`/`matmul` for speed; avoid for‑loops for tensor math.

---

## Tensor Aggregation (min, max, mean, sum)

Aggregation reduces many numbers down to a few (or one):

```python
x = torch.arange(0, 100, 10)   # tensor([ 0, 10, 20, ..., 90])
print(torch.min(x), x.min())   # tensor(0) tensor(0)
print(torch.max(x), x.max())   # tensor(90) tensor(90)

# Mean requires floating point dtype
print(x.dtype)                 # torch.int64 (long)
print(torch.mean(x.float()))   # tensor(45.)

print(torch.sum(x), x.sum())   # tensor(450) tensor(450)
```

Common pitfall: `torch.mean` needs a floating type (e.g., `float32`). Convert with `.float()` or `.type(torch.float32)`.

In a nutshell: min/max/sum work on ints; mean needs floats—convert with `.float()` first.

---

## Positional Min/Max (argmin, argmax)

Sometimes you need where the extreme occurs, not the value itself.

```python
x = torch.arange(1, 100, 10)    # 1, 11, 21, ..., 91
imin = torch.argmin(x)          # index of smallest value (0)
imax = torch.argmax(x)          # index of largest value (9)
print(x[imin], x[imax])         # tensor(1) tensor(91)
```

This is very handy after softmax for picking the predicted class index.

In a nutshell: `argmin/argmax` give positions (indices); index back into the tensor to get the value.

---

## Reshaping, Views, Stacking, Squeeze/Unsqueeze, Permute

These are core tools for fixing shape mismatches and preparing data.

### Reshape vs View
- `reshape` returns a tensor with a new shape (compatible element count).
- `view` returns a new view on the same memory (changing one can change the other).

```python
x = torch.arange(1, 10)           # shape: [9]
xr = x.reshape(1, 9)              # [1, 9]
z = x.view(1, 9)                  # shares memory with x
z[0, 0] = 999                     # modifies x too!
print(x[0])                       # tensor(999)
```

In a nutshell: `reshape` changes shape; `view` is a new window on the same data.

### Stacking
Combine tensors along a new dimension.

```python
x = torch.arange(5)               # [0,1,2,3,4]
stack0 = torch.stack([x, x, x], dim=0)  # shape: [3, 5]
stack1 = torch.stack([x, x, x], dim=1)  # shape: [5, 3]
```

In a nutshell: `stack` adds a new axis and packs tensors along it.

### Squeeze and Unsqueeze
- `squeeze` removes dimensions of size 1.
- `unsqueeze` adds a size‑1 dimension at a given position.

```python
x = torch.randn(1, 9)
xs = x.squeeze()                  # shape: [9]
xu0 = xs.unsqueeze(0)             # shape: [1, 9]
xu1 = xs.unsqueeze(1)             # shape: [9, 1]
```

In a nutshell: Squeeze shrinks away 1‑dims; unsqueeze inserts them where you need.

### Permute (Reorder Dimensions)
Change axis order—common for images.

```python
# Image as HWC (height, width, channels)
img_hwc = torch.rand(224, 224, 3)
# Convert to CHW (channels, height, width)
img_chw = img_hwc.permute(2, 0, 1)   # [3, 224, 224]
```

Note: `permute` returns a view; data is the same, axis order changes.

In a nutshell: `permute` reorders axes, e.g., HWC ↔ CHW for images.

---

## 1) Permute returns a view (memory is shared)

When you call `permute`, you don’t copy the data — you create a different “view” onto the same memory. The numbers are the same; only the dimension order changes.

Example: HWC (height, width, channels) → CHW (channels, height, width) for images.

```python
import torch

X_original = torch.randn(2, 3, 4)      # e.g., NCHW-ish shape for demo
X_permuted = X_original.permute(1, 0, 2)

print("original shape:", X_original.shape)  # torch.Size([2, 3, 4])
print("permuted shape:", X_permuted.shape)  # torch.Size([3, 2, 4])
```

Because `permute` returns a view, in‑place edits to the underlying storage are visible from both tensors.

```python
# Change a single value in-place
X_original[0, 0, 0] = 999
print(X_original[0, 0, 0].item())  # 999.0
print(X_permuted[0, 0, 0].item())  # 999.0  ← same memory
```

- `permute` changes axis order but not the data values.
- A “view” shares the same storage; a “copy” has separate storage.

In a nutshell: `permute` only reorders axes; it returns a view that shares memory with the original tensor.

---

## 2) Indexing multi‑dimensional tensors (with examples)

Indexing lets you select elements across dimensions. Think: outer → inner brackets left to right.

We’ll create a simple shaped tensor to practice: `1 × 3 × 3`.

```python
import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x)
# tensor([[[1, 2, 3],
#          [4, 5, 6],
#          [7, 8, 9]]])
print(x.shape)  # torch.Size([1, 3, 3])
```

Basic indexing:
- `x[0]` → selects the first block (the outermost dimension).
- `x[0, 0]` → selects the first row of that block.
- `x[0, 0, 0]` → selects the first element of that row (value `1`).

Indexing with “all” along a dimension uses a colon `:` (not a semicolon):

```python
x[ :, 0,  : ]   # all blocks, first row, all columns → tensor([[1, 2, 3]])
x[ :, :,  1 ]   # all blocks, all rows, column index 1 → tensor([[2, 5, 8]])
```

Challenges (and solutions):
- “Return 9” → `x[0, 2, 2]`
- “Return 3, 6, 9” as a 1D slice → `x[0, :, 2]` → `tensor([3, 6, 9])`

Tip: When an index is out of range (e.g., using `1` on a size‑1 dimension), you’ll get “index out of bounds” errors. Check `x.shape` first.

In a nutshell: Use commas to step through dimensions and `:` to select all along a dimension; always sanity‑check shapes before indexing.

---

## 3) NumPy ↔ PyTorch interop (dtypes and memory sharing)

You’ll often move data between NumPy and PyTorch.

- NumPy → PyTorch: `torch.from_numpy(ndarray)`
- PyTorch → NumPy: `tensor.numpy()`

Data types (defaults):
- NumPy default float dtype is `float64`.
- PyTorch default float dtype is `torch.float32`.

```python
import numpy as np
import torch

arr = np.arange(1, 9, dtype=np.float64)
t = torch.from_numpy(arr)          # shares memory with arr
print(t.dtype)  # torch.float64 (reflects NumPy dtype)

# Convert dtype if needed
t32 = t.to(torch.float32)
print(t32.dtype)  # torch.float32
```

Memory sharing rules (important):
- `torch.from_numpy(ndarray)` → the returned tensor shares memory with the NumPy array.
  - In‑place edits reflect both ways:
    - `arr += 1` will also change `t`.
    - `t.add_(1)` will also change `arr`.
  - But rebinding creates new storage and does not reflect:
    - `arr = arr + 1` creates a new array; `t` won’t change.
- `tensor.numpy()` → the returned NumPy array shares memory with the tensor (if on CPU).
  - In‑place tensor edits like `tensor.add_(1)` change the array.
  - But `tensor = tensor + 1` rebinds to a new tensor; the array won’t change.

```python
# Demonstrate in-place vs rebinding
arr = np.arange(4.0)
t = torch.from_numpy(arr)
arr += 1            # in-place → reflected in t
print(t)            # tensor([1., 2., 3., 4.], dtype=torch.float64)

arr = arr + 1       # new array → not reflected in t
print(t)            # still tensor([1., 2., 3., 4.], dtype=torch.float64)

# Tensor → NumPy (CPU only)
t_cpu = torch.tensor([1., 2., 3.], dtype=torch.float32)
arr2 = t_cpu.numpy()
t_cpu.add_(10)      # in-place → reflected in arr2
print(arr2)         # [11. 12. 13.]
```

In a nutshell: Interop shares memory; use in‑place ops (`+=`, `add_`) to see changes reflected; plain `x = x + 1` makes a new object.

---

## 4) Reproducibility with random seeds

Neural nets start from random weights. To make experiments repeatable, set a random seed.

```python
import torch

torch.manual_seed(42)           # CPU RNG
# If using CUDA, you may also set CUDA RNGs
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Two random tensors created after seeding will be the same each run
A = torch.rand(3, 4)
B = torch.rand(3, 4)
print(A[:1])
print(B[:1])
```

Notes:
- Seed once before generating random numbers you want to reproduce.
- If you re‑seed in between calls, you’ll “restart” the RNG stream.
- Full determinism across backends can require more settings (e.g., `torch.use_deterministic_algorithms(True)`), but for fundamentals, seeding is usually sufficient.

Make two sequential random calls identical (common in notebooks):

```python
import torch

# Without reseeding: these will differ
torch.manual_seed(42)
A = torch.rand(3, 4)
B = torch.rand(3, 4)
print(torch.allclose(A, B))  # False

# With reseeding before each call: these will match
torch.manual_seed(42)
C = torch.rand(3, 4)
torch.manual_seed(42)
D = torch.rand(3, 4)
print(torch.allclose(C, D))  # True
```

In a nutshell: Call `torch.manual_seed(…)` (and CUDA seeds if needed) to make random tensors repeatable.

---

## 5) Running on the GPU (and how to check)

GPUs speed up tensor math massively. In Colab, you can enable a GPU in the Runtime settings; locally you’ll need a compatible NVIDIA GPU and CUDA drivers per the PyTorch install guide.

Quick checks:

```python
import torch
print(torch.cuda.is_available())   # True if a CUDA GPU is visible to PyTorch
print(torch.cuda.device_count())   # Number of GPUs
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Why it matters:
- Keep tensors and models on the same device.
- Moving data CPU↔GPU is relatively slow; minimize transfers inside tight loops.

In a nutshell: Use a CUDA‑enabled GPU when available; it makes training and inference much faster.

---

## 6) Device‑agnostic code and moving tensors/models

Write code that works on CPU and GPU without modification by picking a target device once and reusing it.

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tensors
x = torch.tensor([1., 2., 3.], device=device)
# or: x = torch.tensor([1., 2., 3.]).to(device)

# Models
import torch.nn as nn
model = nn.Linear(3, 2).to(device)

# Forward pass keeps everything on the same device
out = model(x)
print(out.device)  # should be cuda:0 if GPU available, else cpu
```

Converting to NumPy requires CPU:

```python
# If x is on GPU, .numpy() will raise an error; bring it back first
x_cpu = x.detach().cpu()   # detach if it has grad history
arr = x_cpu.numpy()
```

In a nutshell: Pick a `device` once, move tensors/models to it, and come back to CPU before calling `.numpy()`.

---

## Practice prompts

- Use indexing to return the value `9` and the vector `[3, 6, 9]` from `x = torch.arange(1, 10).reshape(1, 3, 3)`.
- Demonstrate memory sharing by creating a NumPy array, converting with `torch.from_numpy`, and then modifying values in‑place on either side.
- Seed the RNG and generate two identical random tensors; then change the seed and generate a different one.
- Move a tensor and a simple `nn.Linear` model to the GPU (if available), do a forward pass, then bring the output back to CPU and convert to NumPy.

In a nutshell: Small, hands‑on experiments build intuition faster than reading alone — try the prompts above.

---

## Extra resources

- PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
- CUDA semantics & best practices: https://pytorch.org/docs/stable/notes/cuda.html
- NumPy↔PyTorch bridge: https://pytorch.org/docs/stable/notes/interop.html

In a nutshell: The official docs have concise, practical guidance when you want to dive deeper.


