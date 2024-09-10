# Neural Network Optimization with Momentum Methods

## Project Overview
This project, implemented in Python, focuses on exploring various momentum-based methods in neural network training. Our model allows for the selection of layers, activation functions, and L2 regularization. The optimization methods include:

- Gradient Descent (GD)
- Stochastic Gradient Descent (SGD)
- Classical Momentum (CM)
- Nesterov Accelerated Gradient (NAG)

The project also includes a grid search and k-fold cross-validation for parameter tuning and model validation. The performance of the models was tested on MONK datasets and the CUP dataset.

## Objectives
The main objectives of this project are:
1. **Explore momentum-based methods**: Study and compare various optimization techniques.
2. **Validate models**: Perform parameter tuning using grid search and validate using k-fold cross-validation.
3. **Test on MONK and CUP datasets**: Analyze model performance on both simple (MONK) and complex (CUP) datasets.

## Methodology
- **Nested Grid Search**: We performed an initial broad grid search, followed by a refined search in a smaller parameter range.
- **Parallelization**: Used `Joblib` for parallelizing grid search and speeding up cross-validation.
- **Preprocessing**: One-hot encoding was applied for MONK tasks, and a correlation analysis was conducted on the MONK datasets.

## Hardware
Experiments were conducted on a PC with:
- **32 GB RAM**
- **Ryzen 7 5800x 8-core/16-thread processor**

## Key Results
### MONK Datasets:
- Achieved 100% accuracy on MONK1 and MONK2.
- MONK3 reached 98% accuracy.

### CUP Dataset:
- **Final Model**: 
  - Topology: [10, 32, 64, 32, 3]
  - Hidden activation: Sigmoid
  - Output activation: Linear
  - Optimizer: Nesterov Accelerated Gradient (NAG)
  - MSE on Holdout Test: 0.285 ± 0.063

## Conclusion
- **NAG** provides better generalization and stability over classical momentum, especially in complex datasets like CUP.
- Momentum scheduling significantly reduces the need for extensive parameter tuning.

## References
1. Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the Importance of Initialization and Momentum in Deep Learning. Proceedings of the 30th International Conference on Machine Learning, 28(3):1139-1147. 
2. Yang, T., Lin, Q., & Li, Z. (2016). Unified Convergence Analysis of Stochastic Momentum Methods for Convex and Non-Convex Optimization. Available from: [arXiv](https://arxiv.org/abs/1604.03257).
3. Joblib Documentation: [Joblib](https://joblib.readthedocs.io/en/stable/)

