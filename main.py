import argparse
from scipy.io import loadmat
import numpy as np

from video_maker import make_video
from rpca import rpca

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", default=0.05, type=float)
    parser.add_argument("--rank", default=10, type=int)
    parser.add_argument("--tol", default=1e-7, type=float)
    parser.add_argument("--max_iter", default=1000, type=int)

    config = parser.parse_args()
    print(config)
    
    # load the .mat file containing video frames
    mat_file = loadmat('./data/shoppingmall.mat')

    # extract the video frames from the .mat file
    X = mat_file['shoppingmall']

    make_video(X, filename='./output/original.mp4')

    L, S = rpca(X, 1/np.sqrt(max(X.shape)), config.rank, config.tol, config.max_iter)

    make_video(L, filename=f'./output/background_rank{config.rank:0>2}.mp4')
    make_video(S, filename=f'./output/foreground_rank{config.rank:0>2}.mp4')
