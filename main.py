import argparse
from scipy.io import loadmat

from video_maker import make_video
from rpca import rpca

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", default=None, type=float,
                        help='The importance given to sparsity.')
    parser.add_argument("--mu", default=None, type=float,
                        help='The initial value of the penalty parameter in the \
                            Augmented Lagrangian Multiplier (ALM) algorithm.')
    parser.add_argument("--max_iter", default=1000, type=int,
                        help='The maximum number of iterations the optimization algortihm will run for.')
    parser.add_argument("--eps_primal", default=1e-7, type=float,
                        help='The threshold for the primal error in the convex optimization problem.')
    parser.add_argument("--eps_dual", default=1e-5, type=float,
                        help='The theshold for the dual error in the convex optimzation problem.')
    parser.add_argument("--rho", default=1.6, type=float,
                        help='The ratio of the paramter mu between two successive iterations.')
    parser.add_argument("--initial_sv", default=10, type=int,
                        help='The number of singular values to compute during the first iteration.')
    parser.add_argument("--max_mu", default=1e6, type=float,
                        help='The maximum value that mu is allowed to take.')
    parser.add_argument('--verbose', type=str2bool, default=True, 
                        help='Output learning process, both printing and tensorboard.')
    parser.add_argument("--save_interval", default=5, type=int,
                        help='Frequency of saving videos.')

    conf = parser.parse_args()
    print(conf)
    
    # load the .mat file containing video frames
    mat_file = loadmat('./data/shoppingmall.mat')

    # extract the video frames from the .mat file
    X = mat_file['shoppingmall']

    make_video(X, filename='./output/original.mp4')

    L, S, r = rpca(X, conf.lam, conf.mu, conf.max_iter, conf.eps_primal, conf.eps_dual, 
                    conf.rho, conf.initial_sv, conf.max_mu, conf.verbose, conf.save_interval)

    make_video(L, filename=f'./output/background.mp4')
    make_video(S, filename=f'./output/foreground.mp4')
    print(f'Final rank: {r}')
    print('DONE!')
