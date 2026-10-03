# NeuralNet Studio: Efficient Learning via Forgetting Theory

**NeuralNet Studio** is an interactive web dashboard designed to visualize, monitor, and optimize deep learning models. This project specifically explores the integration of **Forgetting Theory** (inspired by cognitive psychology and the Ebbinghaus forgetting curve) into neural network optimization to improve generalization, control weight dynamics, and combat catastrophic forgetting or severe overfitting.

The application combines a powerful **PyTorch** backend for model manipulation with a clean, dynamic frontend built using **Flask** and **Chart.js** to track training and validation metrics in real time.

---

## 🚀 Key Features

* **Real-time Training Diagnostics:** Interactive visualization of Training vs. Validation loss curves to instantly catch overfitting thresholds.
* **Cognitive Forgetting Integration:** Built-in experimentation frameworks for simulating memory decay, structural pruning, and adaptive regularization based on forgetting dynamics.
* **On-the-Fly Configuration:** Dynamically tweak configurations like total epochs, learning rates, and batch sizes before initializing training loops.
* **Matplotlib Export:** One-click visual extraction of standard publication-ready loss curves for academic documentation.
* **Instant Inference Engine:** An built-in "Predict" panel to test localized model inferences instantly post-training.

---

## 🧠 Forgetting Theory Implementation

In traditional training configurations, networks frequently over-optimize on niche training data noise—visible when training loss plummets while validation loss climbs. This project introduces engineered "forgetting" dynamics to resolve this bottleneck:

* **Weight Decay & Memory Fade:** Using controlled L2 penalties to continuously decay low-impact parameters, effectively treating minor noise as forgettable artifacts.
* **Elastic Overwrite Protection:** Implementing consolidation metrics to ensure core generalized patterns are retained while redundant paths are recycled.
* **Synaptic Dropout Schedules:** Leveraging adaptive dropout constraints to prevent static co-adaptation of features.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask, PyTorch (NoisyLinear, Leaky ReLU, Adam Optimizer)
* **Frontend:** HTML5, CSS3, JavaScript, Chart.js, TailwindCSS / Custom Theme
* **Logging/Plotting:** Matplotlib
