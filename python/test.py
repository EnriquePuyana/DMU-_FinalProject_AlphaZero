#%% File
import numpy as np
import chess 
import chess_api
from ChessNN import NNetWrapper
import scipy.sparse as nps
import MCTS
import copy
import cProfile
import pstats
import multiprocessing
import time
import util
import random
import pickle
import matplotlib.pyplot as plt
plt.style.use('dark_background')

#%% Setup
draw_fens = [
    '1rk5/Q1q5/KR6/8/8/8/8/8 w - - 1 1',
    '1qk5/R1r5/KQ6/8/8/8/8/8 b - - 1 1',
    ]
winning_fens = [
    'Rqk5/r7/KQ6/8/8/8/8/8 w - - 1 1',
    'q1k5/Q1r5/K7/8/8/8/8/8 b - - 1 1',
    ]
train_boards = [chess_api.BoardWrapper_3x3(fen) for fen in draw_fens]
eval_boards = [chess_api.BoardWrapper_3x3(fen) for fen in winning_fens]
eval_boards.extend(train_boards)

load_data = True
load_number = 60
nnet = NNetWrapper(three_by_three=True)
if load_data:
    nnet.load_checkpoint(folder='board_3x3', filename=f"checkpoint_mcts25_eps50_fen2_{load_number}")
    with open(f'/home/mapu3213/gitrepos/DMU-_FinalProject_AlphaZero/data/eval_data_mcts25_eps50_fen2_{load_number}.pkl', 'rb') as file:
            # Load the data from the pickle file
            pickle_data = pickle.load(file)

args = {
    # 'numIters': 1000,
    'numEps': 100,                          # Number of complete self-play games to simulate during a new iteration.
    'numTrainingEps': 100000,                  # Number of times to run numEps
    'tempThreshold': 15,                    # When to start always playing the best action
    # 'updateThreshold': 0.55,              # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxBufferSize': 100000,                # Number of game examples to train the neural networks.
    'trainBatchSize': 1000,                 # Number of experience tuples to train the network on
    'numMCTSSims': 25,                      # Number of games moves for MCTS to simulat 
    'MCTSThinkTime': None,                  # Think time # Not in use
    'arenaCompare': 40,                     # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 4,                             # Exploration bonus
    # 'numItersForTrainExamplesHistory': 20,  # 
    'discount': 1,                          # Discount factor for use in MCTS
    'maxProcesses': 100,                     # Number of threads to use
    'start2train': 0                        # Whento start training if not waiting for buffer to fill up
}

mcts_eng = MCTS.MCTSEngine(nnet, args)

print(nnet.predict(eval_boards[0]))