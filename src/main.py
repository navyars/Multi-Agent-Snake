import argparse

import ActorCritic
import AsynchronousQ
import Constants

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', type=str,
                        choices=['train', 'simulate'],
                        required=True,
                        help="""Sets mode of execution.
                                If train, then the RL agents are trained using
                                specified algorithm and number of time_steps.
                                If simulate, then an episode is simulated using
                                pre-trained agents.""")
    parser.add_argument('--train_time_steps', type=int, default=10000,
                        help="The number of time steps to run training for. Requires: --mode=train.")
    parser.add_argument('--algorithm', type=str, choices=['asyncQ', 'actorcritic'], required=True,
                        help="The algorithm to be used for training.")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help="The directory to store all checkpoints during training. Requires: --mode=train.")
    parser.add_argument('--checkpoint_frequency', type=int, default=500,
                        help="The number of iterations after which to periodically checkpoint the weights. Requires: --mode=train.")

    parser.add_argument('--play', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="Setting this allows user to play the game alongside RL agents. Requires: --mode=simulate.")
    parser.add_argument('--trained_dir', type=str, default='checkpoints',
                        help="Directory to load checkpoints/numpy files from.")
    parser.add_argument('--trained_ckpt_index', type=int, default=-1,
                        help="The checkpoint number that refers the file to be loaded.")

    parser.add_argument('--use_relative_state', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="States are defined either absolutely or relative to frame of reference of snake. Setting this to true chooses the latter.")
    parser.add_argument('--multi_agent', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="State definition matters on also whether it's a multi-agent scenario or single-agent.")
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest food points to be stored as part of the state.")
    parser.add_argument('--gamma', type=float, default=1.0,
                        help="Gamma value used for calculating returns.")

    parser.add_argument('--alphaW', type=float, default=0.0022,
                        help="Learning rate for value function. Requires: --algorithm=actorcritic")
    parser.add_argument('--alphaTheta', type=float, default=0.0011,
                        help="Learning rate for policy function. Requires: --algorithm=actorcritic")
    parser.add_argument("--y", type=float, default=1.0,
                        help="Gamma value for calculating returns.")

    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate for training neural network approximator. Requires: --algorithm=asyncQ")

    args = parser.parse_args()

    if args.mode=='train':
        load = (args.trained_ckpt_index != -1)
        if args.algorithm == 'asyncQ':
            AsynchronousQ.train(max_time_steps=args.train_time_steps, reward=1, penalty=-10, asyncUpdate=30, globalUpdate=120, relativeState=args.use_relative_state,
                                                    checkpointFrequency=args.checkpoint_frequency, checkpoint_dir=args.checkpoint_dir,
                                                    load=load, load_dir=args.trained_dir, load_time_step=args.trained_ckpt_index)
        else:
            ActorCritic.train(Constants.gridSize, args.use_relative_state, args.multi_agent, args.k, args.alphaTheta, args.alphaW, args.gamma, args.train_time_steps,
                                                    checkpointFrequency=args.checkpoint_frequency, checkpoint_dir=args.checkpoint_dir,
                                                    load=load, load_dir=args.trained_dir, load_time_step=args.trained_ckpt_index)
        print("Training complete.")
    else:
        if args.algorithm == 'asyncQ':
            AsynchronousQ.graphical_inference(Constants.gridSize, args.use_relative_state, args.multi_agent, args.k,
                                                                        load_dir=args.trained_dir, load_time_step=args.trained_ckpt_index, play=args.play, scalingFactor=9)
        else:
            ActorCritic.graphical_inference(Constants.gridSize,  args.use_relative_state, args.multi_agent, args.k,
                                                                    load_dir=args.trained_dir, load_time_step=args.trained_ckpt_index, play=args.play, scalingFactor=9)
        print("Inference complete.")

if __name__ == '__main__':
    main()
