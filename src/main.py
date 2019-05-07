import argparse

from Algorithm import actorCritic
from AsychronousRoughAlgo import mainAlgorithm as asyncQ
import Constants

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_episode_length', type=int, default=300,
                        help="The maximum time that a single game is allowed to proceed.")
    parser.add_argument('--num_snakes', type=int, default=2,
                        help="The number of snake agents")
    parser.add_argument('--min_food', type=int, default=10,
                        help="The minimum number of food points that should be present at all times.")
    parser.add_argument('--grid_size', type=int, default=50,
                        help="Defines the square grid size of the game's arena.")

    parser.add_argument('--mode', type=str,
                        choices=['train', 'simulate'], default='train',
                        required=True,
                        help="""Sets mode of execution.
                                If train, then the RL agents are trained using
                                specified algorithm and number of time_steps.
                                If simulate, then an episode is simulated using
                                pre-trained agents.""")
    parser.add_argument('--train_time_steps', type=int, default=10000000,
                        help="The number of time steps to run training for. Requires: --mode=train.")
    parser.add_argument('--algorithm', type=str, choices=['asyncQ', 'actorcritic'], default='asyncQ',
                        help="The algorithm to be used for training.")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help="The directory to store all checkpoints during training. Requires: --mode=train.")

    parser.add_argument('--graphics', type=bool, default=False,
                        help="Whether to run graphics mode or text mode for simulation. Requires: --mode=simulate.")
    parser.add_argument('--play', type=bool, default=False,
                        help="Setting this allows user to play the game alongside RL agents. Requires: --mode=simulate, --graphics=True.")
    parser.add_argument('--trained_dir', type=str, default='checkpoints',
                        help="Directory to load checkpoints/numpy files from.")

    parser.add_argument('--use_relative_state', type=bool, default=False,
                        help="States are defined either absolutely or relative to frame of reference of snake. Setting this to true chooses the latter.")

    args = parser.parse_args()

    Constants.numberOfSnakes = args.num_snakes
    Constants.gridSize = args.grid_size
    Constants.globalEpisodeLength = args.max_episode_length
    Constants.maximumFood = args.min_food

    if args.mode=='train':
        if args.algorithm == 'asyncQ':
            asyncQ(max_time_steps=args.train_time_steps, reward=1, penalty=-10, asyncUpdate=20, globalUpdate=60)
        else:
            actorCritic(args.grid_size, args.use_relative_state, (args.num_snakes > 1), 3, 0.01, 0.01, 0.99, args.train_time_steps)
        print("Training complete.")

if __name__ == '__main__':
    main()
