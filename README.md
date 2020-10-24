# Comparative Evaluation of Feature Descriptors Through Bag of Visual Features with Support Vector Machine on Embedded GPU System

![GitHub repo size](https://img.shields.io/github/repo-size/whoisraibolt/BoVF-with-SVM-Classifier)
![GitHub](https://img.shields.io/github/license/whoisraibolt/BoVF-with-SVM-Classifier)

## Overview

Comparative evaluation of Local Feature Descriptors such as SIFT, SURF, and KAZE, and Local Binary Descriptors such as BRIEF, ORB, BRISK, AKAZE, and FREAK through Bag of Visual Features (BoVF) approach with Support Vector Machine (SVM) classifier for the tasks of recognition and classification on six visual datasets (MNIST, JAFFE, Extended CK+, FEI, CIFAR-10, and FER-2013) on an embedded GPU system: The NVIDIA's Jetson Nano.
-->

**Notice:**

- The proposed work was implemented in Python (version 3.6) as well as OpenCV (version 4.1) using a GPU ARM architecture and might not work with other versions.

- In this repository, only the directory with the MNIST, CIFAR-10, and FER-2013 visual datasets are available. The directory where the other visual datasets should stay is not available in this GitHub repository since it would violate the dataset rules.

- To download and use other visual datasets presented here, we strongly recommend reading the use of the norms of them. However, we recommend reading the use of the norms of the CIFAR-10, FER-2013, and MNIST as well.

## Dependencies

To install the dependencies run:

`pip install -r requirements.txt`

## Usage

`python main.py --detector <detector> --descriptor <descriptor> --dataset <dataset>`

| Arguments     | Info                                                                    |
| :------------ | :---------------------------------------------------------------------- |
| `-h`, `--help`| Show help message and exit                                              |
| `--detector`  | Specify SIFT or SURF or KAZE or ORB or BRISK or AKAZE                   |
| `--descriptor`| Specify SIFT or SURF or KAZE or BRIEF or ORB or BRISK or AKAZE or FREAK |
| `--dataset `  | Specify MNIST or JAFFE or Extended-CK+ or FEI or CIFAR-10 or FER-2013   |

## Examples

####  Help
`python main.py --help`

#### Evaluation of BRIEF Descriptor on CIFAR-10 visual datasets
`python main.py --detector ORB --descriptor BRIEF --dataset CIFAR-10`

## Credits

- [Mlubega](https://github.com/mlubega/cv "Mlubega")

- [Fast.ai](https://forums.fast.ai/t/lesson-6-advanced-discussion/31442/3 "Fast.ai")

- [Scikit-learn.org](https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html "Scikit-learn.org")

## License

Code released under the [MIT](https://github.com/whoisraibolt/BoVF-with-SVM-Classifier/blob/master/LICENSE "MIT") license.
