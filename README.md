# Deep Learning with PyTorch

My step‑by‑step notes and practice code while learning Deep Learning with PyTorch. This repo includes:

- Concise, beginner‑friendly notes (Markdown)
- Executable Jupyter notebooks with code experiments
- Small projects/demos as I progress (e.g., MNIST, Food101)

The aim is to learn by doing: write small bits of code, visualize results, and summarize key ideas.

Quick link: See consolidated chapter notes in [chapter_0_fundamental_notes.md](./chapter_0_fundamentals/chapter_0_fundamental_notes.md).

## Repository Structure

```
deep-learning-with-pytorch/
├─ README.md
└─ chapter_0_fundamentals/
	 ├─ chapter_0_fundamental_notes.md   # narrative notes
	 └─ 00_pytorch_fundamentals.ipynb    # hands-on fundamentals notebook
```

As new chapters are added, each will typically contain a notes file and one or more notebooks.

## Environment & Setup

You can run everything locally or in Google Colab.

- Local (VS Code recommended):
	- Install Python 3.10+ and pip
	- Create a virtual environment (optional but recommended)
	- Install PyTorch following the instructions for your OS/GPU: https://pytorch.org/get-started/locally/
	- Install Jupyter support: `pip install jupyter matplotlib pandas numpy`

- Google Colab:
	- Open the notebook in Colab
	- Runtime → Change runtime type → set Hardware accelerator to GPU (optional)

## How to Use

1) Read the Markdown notes in each chapter to get the concepts.
2) Open the paired notebook to run code, tweak values, and explore shapes/dtypes/devices.
3) Use the notes as a quick reference while experimenting in the notebook.

Tip: If you hit errors like “mat1 and mat2 shapes cannot be multiplied”, check tensor shapes and transpose/reshape as needed.

## Contributing / Feedback

Suggestions, fixes, and learning tips are welcome. Feel free to open an issue or submit a PR with a short description of the change.

## References

- PyTorch Docs: https://pytorch.org/docs/stable/
- Learn PyTorch (book/site): https://www.learnpytorch.io/
- PyTorch Forums: https://discuss.pytorch.org/

