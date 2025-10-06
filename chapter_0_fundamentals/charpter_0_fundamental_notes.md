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
- **Simple rules work well:** If a few clean rules solve it, don’t overcomplicate things.
- **Zero tolerance for errors:** DL makes probabilistic guesses. If errors are unacceptable, a deterministic rules system may be safer.
- **Too little data:** DL usually needs lots of data (though transfer learning can help).

**In a nutshell:** Avoid DL when you need transparency, simple rules suffice, errors are unacceptable, or data is scarce.

---

## Structured vs Unstructured Data
- **Structured data** looks like tables, rows and columns with numbers or categories (think spreadsheets or databases). Traditional ML methods often work best here.
- **Unstructured data** includes images, free‑text, audio, and video. This is where deep learning really shines because neural networks can learn rich features automatically.

**In a nutshell:** Tables → traditional ML. Images/text/audio → deep learning.

---

## Algorithms: ML vs DL
**Common traditional ML algorithms (great for structured data):**
- Gradient Boosted Trees (e.g., XGBoost)
- Random Forests
- Support Vector Machines (SVM)
- k‑Nearest Neighbors (kNN)
- Naive Bayes

**Common deep learning architectures (great for unstructured data):**
- **MLP / Fully Connected Networks:** general‑purpose neural nets.
- **CNNs:** excel at images by learning spatial patterns.
- **RNNs / sequence models:** handle ordered data like time series or text.
- **Transformers:** state‑of‑the‑art for language and increasingly for vision, audio, and multimodal tasks.

**In a nutshell:** Use boosted trees/forests for tables; use CNNs/RNNs/Transformers for images, sequences, and text.

---

## Shallow vs Deep
Traditional ML models are sometimes called “shallow” because they apply fewer layers of transformation to the data. Deep learning stacks many layers, letting the model build up complex representations step by step (edges → shapes → parts → objects, for example).

**In a nutshell:** “Deep” means many layers that learn increasingly abstract features.

---

## Neural Networks (Basics)
A neural network is a stack of layers: **input → hidden layers → output**. Each layer contains “neurons” (units) that do small calculations. During training, the network adjusts its **weights and biases** so that outputs become more accurate over time.

Key ideas:
- **Representations:** The network learns useful features on its own (you don’t hand‑engineer them).
- **Non‑linearities:** Functions like ReLU or sigmoid let networks model curved, complex relationships, not just straight lines.
- **Depth & width:** You can choose how many layers (depth) and how many units per layer (width) to suit the problem.

Example: In computer vision, architectures like ResNet‑34 and ResNet‑152 illustrate “depth” (the number of layers) in practice.

**In a nutshell:** Neural nets learn their own features through many layered transformations.

---

## Learning Paradigms

### Supervised Learning
This is the most common type of learning. You have **inputs and their labels**.  
Example: Give the model thousands of cat and dog photos, each labeled “cat” or “dog.” The model learns the mapping between photo → label, so it can predict on new, unlabeled photos.

**In a nutshell:** Labeled data teaches the model what’s what.

---

### Unsupervised & Self-Supervised Learning
- **Unsupervised learning:** You only have data, no labels. The model tries to find hidden patterns or groupings.  
  Example: A folder of cat and dog photos without labels. The model might cluster similar images, even if it doesn’t know the words “cat” or “dog.”

- **Self-supervised learning:** A clever twist where the model generates its own signals from data.  
  Example: Predicting missing words in a sentence. No manual labels needed, data creates its own training signal.

**In a nutshell:** Unsupervised = no labels. Self-supervised = model makes its own labels/signals.

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

## Tensor Data Types (dtype)

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

## Matrix Multiplication Preview

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


