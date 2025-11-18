# score-based-mnist

A way faster and more refined MNIST-Score-Matching-Diffusion implementation based on: https://github.com/mfkasim1/score-based-tutorial

Trains a conditional Score-Matching-Diffusion U-Net in under 10min on a Nvidia RTX 3070 using Stochastic Weight Averaging with EMA and different schedulers for time step sampling and, batch size. learning rate and beta1.

Final training results:

![MNIST Results](./images/samples_final.png "MNIST Results")
