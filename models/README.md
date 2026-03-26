# Models Directory

This directory is intended to store the PyTorch model weights (`.pth` files) generated during training.

**Note on Version Control:** 
Model weight files (especially for HuBERT, which is >350 MB) are too large to be tracked directly by Git and are therefore included in the `.gitignore` file.

### How to get the models
To generate the model weights, run the `notebooks/final_notebook.ipynb` end-to-end (preferably on Kaggle with a T4 GPU). The following checkpoints will be saved in the `/kaggle/working` directory:
- `best_cnn.pth` (Milestone 3)
- `best_crnn.pth` (Milestone 4)
- `best_hubert.pth` (Milestone 5)

You can download these files from the Kaggle output section and manually place them in this `/models/` directory for local inference testing.
