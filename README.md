# Robust PCA for Moving Object Detection in Video

This project demonstrates the use of Robust Principal Component Analysis (RPCA) for detecting and removing moving objects from a video sequence. The RPCA algorithm decomposes a video sequence into a low-rank background matrix and a sparse moving object matrix, allowing for the detection and removal of moving objects from the background.

## Requirements

- Python 3.x
- OpenCV (version 4.x or later)
- NumPy
- Scikit-learn
- Tensorboard (optional, for learning process visualization)
- Matplotlib (optional, for result visualization)

## Usage

1. Clone or download the repository to your local machine.

2. Install the required packages using pip or conda:

```
pip install opencv-python numpy scikit-learn tensorboard matplotlib
```

3. Download the input video file `shoppingmall.mat` from the [Google Drive](https://drive.google.com/file/d/1CuVAG3uWnwq6QmI3vARUizOF01Ubfz9k/view?usp=sharing) link. Place it in the `data` folder.

4. To use this project, simply run the `main.py` script with the desired command line arguments. The available command line arguments are:

- `--lambda`:       regularization parameter for the sparse component.
- `--mu`:           initial value of the penalty parameter in the Augmented Lagrangian Multiplier (ALM) algorithm.
- `--max_iter`:     maximum number of iterations for the optimization algorithm (default: 1000).
- `--eps_primal`:   threshold for the primal error in the convex optimization problem (default: 1e-7).
- `--eps_dual`:     theshold for the dual error in the convex optimzation problem (default: 1e-5).
- `--rho`:          ratio of the paramter mu between two successive iterations (default: 1.6).
- `--initial_sv`:   number of singular values to compute during the first iteration (default: 10).
- `--max_mu`:       maximum value that mu is allowed to take. (default: 1e6).
- `--verbose`:      whether output learning process, both printing and tensorboard (default: True).
- `--save_interval`:number of iterations for which the result will be saved (default: 10).


For example, to run the script with a regularization parameter of 0.1 and the max iterations of 20, use the following command:

```
python main.py --lambda 0.1 --max_iter 20
```

or

```
nohup python -u main.py --lambda 0.1 --max_iter 20 > log.out &
```

5. The resulting low-rank and sparse components will be saved as separate videos in the `output` folder.

## Credits

This project is inspired by the papers:

1. Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). Robust principal component analysis?. Journal of the ACM (JACM), 58(3), 1-37.

2. Lin, Z., Chen, M., & Ma, Y. (2010). The augmented lagrange multiplier method for exact recovery of corrupted low-rank matrices. arXiv preprint arXiv:1009.5055.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
