# Project Description: Robust PCA for Moving Object Detection in Video

This project demonstrates the use of Robust Principal Component Analysis (RPCA) for detecting and removing moving objects from a video sequence. The RPCA algorithm decomposes a video sequence into a low-rank background matrix and a sparse moving object matrix, allowing for the detection and removal of moving objects from the background.

The implementation is in Python and uses the NumPy and OpenCV libraries. The main file is `main.py`, which contains the RPCA algorithm implementation and the code to read and write video files. The algorithm is applied to the input video file `shoppingmall.mat` in the directory `data`, and the output video files is written in folder `output`.

## Requirements

- Python 3.x
- OpenCV (version 4.x or later)
- NumPy
- Matplotlib (optional, for visualization)

## Usage

1. Clone or download the repository to your local machine.

2. Install the required packages using pip or conda:

```
pip install opencv-python numpy matplotlib
```

3. Download the input video file `shoppingmall.mat` from the [Google Drive](https://drive.google.com/file/d/1CuVAG3uWnwq6QmI3vARUizOF01Ubfz9k/view?usp=sharing) link. Place it in the `data` folder.

4. To use this project, simply run the `main.py` script with the desired command line arguments. The available command line arguments are:

- `--lambda`: regularization parameter for the sparse component (default: 0.05).
- `--rank`: rank of the low-rank component (default: 10).
- `--tol`: tolerance for convergence of the optimization algorithm (default: 1e-7).
- `--max_iter`: maximum number of iterations for the optimization algorithm (default: 1000).

For example, to run the script with a regularization parameter of 0.1 and a rank of 20, use the following command:

```
python main.py --lambda 0.1 --rank 20
```

5. The resulting low-rank and sparse components will be saved as separate videos in the `output` folder.

## Credits

This project is inspired by the paper "Robust Principal Component Analysis for Background Subtraction: Systematic Evaluation and Comparative Analysis" by Guyon, C., Bouwmans, T., & Zahzah, E. H. (Principal component analysis 10, 2012).

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
