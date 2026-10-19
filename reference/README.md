npgnn.py is a NumPy re-implementation of SpatioTemporalGNN (same layers, PyTorch default
initialisation, Adam, batch 64, 50 epochs, dropout 0.2; 95,236 parameters; gradients
checked by finite differences). It was used only because PyTorch could not be installed
in the review environment. Do not use it for reported results: re-run with the
PyTorch scripts in scratch/. The JSON files are the review's results for comparison.
