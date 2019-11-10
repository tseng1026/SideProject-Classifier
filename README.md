# SideProject - ImageClassifier

## Basic Execution
- **Platform:** Linux (Workstation)
- **Language:** Python3
- **Environment:** GPU
- **Usage:**
	- ``CUDA_VISIBLE_DEVICES=<number> python Train.py -m <model name>``
	- ``CUDA_VISIBLE_DEVICES=<number> python Test.py -m <model name> -o <output file>``
- **Requirements:**
	- python3.6
	- torch 1.2		``pip install torch``
	- torchvision 0.4``pip install torchvision``
	- matplotlib 3.1	``pip install matplotlib``
- **Final Result:** Accuracy on testdata - 0.7023

## Reference
- *ImageNet Classification with Deep Convolutional Neural Networks*, Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, NIPS 2012
