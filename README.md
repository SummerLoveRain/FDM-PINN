# [FDM-PINN: Physics-Informed Neural Network based on Fictitious Domain Method](https://github.com/SummerLoveRain/FDM-PINN)

In this article,  we present a physics-informed neural network combined with  fictitious domain method (FDM-PINN) to study  linear elliptic and parabolic problems with Robin boundary  condition. Our goal here to develop a  deep learning framework where one solves a variant of the original problem on the full $\Omega$, followed by a well-chosen correction on small domain $\omega (\overline{\omega} \subset \Omega)$, which is geometrically simple shaped domain. We study the applicability and accuracy of FDM-PINN for the elliptic and parabolic problems with fixed $\omega$ or moving $\omega$. This method is of the virtual control type and relies on a well designed neural network. Numerical results obtained by FDM-PINN for two-dimensional elliptic and parabolic problems are given, which are more accurate than the results obtained by least-squares/fictitious domain method in reference [9].

For more information, please refer to the following: (https://doi.org/10.1080/00207160.2022.2128674)

## Citation

    @article{yang2023fdm,
    title={FDM-PINN: Physics-informed neural network based on fictitious domain method},
    author={Yang, Qihong and Yang, Yu and Cui, Tao and He, Qiaolin},
    journal={International Journal of Computer Mathematics},
    volume={100},
    number={3},
    pages={511--524},
    year={2023},
    publisher={Taylor \& Francis}
    }

