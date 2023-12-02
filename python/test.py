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
import matplotlib.pyplot as plt
plt.style.use('dark_background')

#%% Setup
draw_fens = [
    '1rk5/Q1q5/KR6/8/8/8/8/8 w - - 1 1',
    '1qk5/R1r5/KQ6/8/8/8/8/8 w - - 1 1',
    ]
winning_fens = [
    'Rqk5/r7/KQ6/8/8/8/8/8 w - - 1 1',
    'q1k5/Q1r5/K7/8/8/8/8/8 b - - 1 1',
    ]
train_boards = [chess_api.BoardWrapper_3x3(fen) for fen in draw_fens]
eval_boards = [chess_api.BoardWrapper_3x3(fen) for fen in winning_fens]
eval_boards.extend(train_boards)

nnet = NNetWrapper(three_by_three=True)
# nnet.load_checkpoint(folder='55conv', filename="checkpoint_5s_c4_1.")

args = {
    'numIters': 1000,
    'numEps': 50,              # Number of complete self-play games to simulate during a new iteration.
    'numTrainingEps': 200  ,     # Number of times to run numEps
    'tempThreshold': 15,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxBufferSize': 10000,    # Number of game examples to train thche neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulat 
    'MCTSThinkTime': None,       # Think time # Not in use
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 4,
    'numItersForTrainExamplesHistory': 20,
    'discount': 0.99,
    'maxProcesses': None
}

mcts_eng = MCTS.MCTSEngine(nnet, args)

# %% Play Epoch
# profiler = cProfile.Profile()
# profiler.enable()


if __name__=='__main__':
    
    # start_time = time.time()
    # ndx_train_step = 0
    # while ndx_train_step<1:
    #     examples, reasons = board.fictitious_epoch(args, nnet, train_step_num=ndx_train_step)
    #     #nnet.train(examples)
    #     # nnet.save_checkpoint(folder='55conv', filename=f'checkpoint_20s_c4_{ndx_train_step}.pth.tar')
    #     ndx_train_step += 1
    # end_time = time.time()
    
    start_time_p = time.time()
    ndx_train_step = 0
    experience_buffer = []
    evals_vs = np.empty((0,len(eval_boards)), dtype=float)
    times = []
    v_colors = ['limegreen', 'dodgerblue', 'red', 'red']
    expected_evals = [1, -1, 0, 0]
    evals_pis_1 = np.empty((0,len(eval_boards[0].list_valid_moves())), dtype=float)
    pi_1_colors = ['red', 'green']
    pi_1_labels = [str(move) for move in eval_boards[0].list_valid_moves()]
    evals_pis_n1 = np.empty((0,len(eval_boards[1].list_valid_moves())), dtype=float)
    pi_n1_colors = ['red', 'red', 'green', 'red', 'red', 'red', 'green']
    pi_n1_labels = [str(move) for move in eval_boards[1].list_valid_moves()]
    while ndx_train_step<args['numTrainingEps']:
        for board in train_boards:
            experience, _ = board.fictitious_epoch_parallel(args, nnet, train_step_num=ndx_train_step)
            experience_buffer = util.append_and_assert_length(experience_buffer, experience, args['maxBufferSize'])
        if len(experience_buffer)==args['maxBufferSize']:
            nnet.train(experience_buffer)
            nnet.save_checkpoint(folder='board_3x3', filename=f'checkpoint_mcts50_eps50_fen2{ndx_train_step}.pth.tar')
            
            evals_prediction = [nnet.get_valid_action_probs(board) for board in eval_boards]
            evals_v = np.array([[v[0] for (_, v) in evals_prediction]], dtype=float)
            evals_vs = np.vstack((evals_vs, evals_v))
            
            evals_pis = [np.array([pi]) for (pi, _) in evals_prediction[0:2]]
            evals_pis_1 = np.vstack((evals_pis_1, evals_pis[0]))
            evals_pis_n1 = np.vstack((evals_pis_n1, evals_pis[1]))
            
            times.append(args['numEps']*(ndx_train_step+1)*len(train_boards))
            
            fig, ax = plt.subplots() 
            # ax.grid(True, linestyle=':')
            for ndx_board in range(len(eval_boards)):
                ax.plot(times, evals_vs[:, ndx_board], marker='.', linestyle='-', color=v_colors[ndx_board])
                ax.plot(times, [expected_evals[ndx_board] for _ in times], marker='', linestyle=':', color=v_colors[ndx_board])
            ax.set_xlabel('Self Play Games')
            ax.set_ylabel('Evaluation')
            ax.set_ylim((-1.1, 1.1))
            fig.set_dpi(500)
            plt.show()
            
            fig, ax = plt.subplots() 
            # ax.grid(True, linestyle=':')
            for ndx_move in range(len(evals_pis_1.transpose())):
                ax.plot(times, evals_pis_1[:, ndx_move], marker='.', linestyle='-', color=pi_1_colors[ndx_move], label=pi_1_labels[ndx_move])
            ax.plot(times, [1 for _ in times], marker='', linestyle=':', color='limegreen')
            ax.set_xlabel('Self Play Games')
            ax.set_ylabel('Probability')
            ax.legend(loc='best', fancybox=True, framealpha=1, handlelength=1)
            ax.set_ylim((0, 1.1))
            fig.set_dpi(500)
            plt.show()
            
            fig, ax = plt.subplots() 
            # ax.grid(True, linestyle=':')
            for ndx_move in range(len(evals_pis_n1.transpose())):
                ax.plot(times, evals_pis_n1[:, ndx_move], marker='.', linestyle='-', color=pi_n1_colors[ndx_move], label=pi_n1_labels[ndx_move])
            ax.plot(times, [1 for _ in times], marker='', linestyle=':', color='limegreen')
            ax.set_xlabel('Self Play Games')
            ax.set_ylabel('Probability')
            ax.legend(loc='best', fancybox=True, framealpha=1, handlelength=1)
            ax.set_ylim((0, 1.1))
            fig.set_dpi(500)
            plt.show()  
            
        ndx_train_step += 1
    end_time_p = time.time()
    print(end_time_p - start_time_p)
        

    # profiler.disable()
    # with open('profile_results.txt', 'w') as f:
    #         stats = pstats.Stats(profiler, stream=f)
    #         stats.sort_stats('tottime')
    #         stats.print_stats()



























