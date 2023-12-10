#%% File
import numpy as np 
import chess_api
from ChessNN_large import NNetWrapper
import MCTS
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
load_number = 39
nnet = NNetWrapper(three_by_three=True)
if load_data:
    nnet.load_checkpoint(folder='board_3x3', filename=f"checkpoint_large_mcts25_eps100_fen2_{load_number}")
    with open(f'/home/mapu3213/gitrepos/DMU-_FinalProject_AlphaZero/data/eval_data_large_mcts25_eps50_fen2_{load_number}.pkl', 'rb') as file:
            # Load the data from the pickle file
            pickle_data = pickle.load(file)

args = {
    # 'numIters': 1000,
    'numEps': 500,                          # Number of complete self-play games to simulate during a new iteration.
    'numTrainingEps': 100000,                  # Number of times to run numEps
    'tempThreshold': 15,                    # When to start always playing the best action
    # 'updateThreshold': 0.55,              # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxBufferSize': 1000000,                # Number of game examples to train the neural networks.
    'trainBatchSize': 10000,                 # Number of experience tuples to train the network on
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
        
    ndx_train_step = load_number + 1
        
    if load_data:
        experience_buffer = pickle_data[0] # [] # pickle_data[0] #
        times = pickle_data[1] # [] # pickle_data[1] #
        evals_vs = pickle_data[2] # np.empty((0,len(eval_boards)), dtype=float) # pickle_data[2] #
        evals_pis_1 = pickle_data[3] # np.empty((0,len(eval_boards[0].list_valid_moves())), dtype=float) # pickle_data[2] #
        evals_pis_n1 = pickle_data[4] # np.empty((0,len(eval_boards[1].list_valid_moves())), dtype=float) # pickle_data[3] #
    else:
        experience_buffer = [] # pickle_data[0] #
        times = [] # pickle_data[1] #
        evals_vs = np.empty((0,len(eval_boards)), dtype=float) # pickle_data[2] #
        evals_pis_1 = np.empty((0,len(eval_boards[0].list_valid_moves())), dtype=float) # pickle_data[2] #
        evals_pis_n1 = np.empty((0,len(eval_boards[1].list_valid_moves())), dtype=float) # pickle_data[3] #
    
    v_colors = ['limegreen', 'dodgerblue', 'red', 'red']
    expected_evals = [1, -1, 0, 0]
    pi_1_colors = ['red', 'limegreen']
    pi_1_labels = [str(move) for move in eval_boards[0].list_valid_moves()]
    pi_n1_colors = ['red', 'red', 'dodgerblue', 'skyblue', 'limegreen', 'skyblue', 'dodgerblue']
    pi_n1_labels = [str(move) for move in eval_boards[1].list_valid_moves()]
    train_delay = 0
    while ndx_train_step<args['numTrainingEps']:
        for board in train_boards:
            experience, _ = board.fictitious_epoch_parallel(args, nnet, train_step_num=ndx_train_step)
            experience_buffer = util.append_and_assert_length(experience_buffer, experience, args['maxBufferSize'])
            train_delay += len(experience)
        if (len(experience_buffer)==args['maxBufferSize'] or ndx_train_step>=args['start2train']) and train_delay>=args['trainBatchSize']:
            train_delay = 0 # Reset train delay
            nnet.train(random.sample(experience_buffer, args['trainBatchSize']))
            
            evals_prediction = [nnet.get_valid_action_probs(board) for board in eval_boards]
            evals_v = np.array([[v[0] for (_, v) in evals_prediction]], dtype=float)
            evals_vs = np.vstack((evals_vs, evals_v))
            
            evals_pis = [np.array([pi]) for (pi, _) in evals_prediction[0:2]]
            evals_pis_1 = np.vstack((evals_pis_1, evals_pis[0])) 
            evals_pis_n1 = np.vstack((evals_pis_n1, evals_pis[1]))
            
            times.append(args['numEps']*(ndx_train_step+1)*len(train_boards))
            
            if ndx_train_step%1==0:
                nnet.save_checkpoint(folder='board_3x3', filename=f'checkpoint_large_mcts25_eps100_fen2_{ndx_train_step}.pth.tar')
                with open(f'/home/mapu3213/gitrepos/DMU-_FinalProject_AlphaZero/data/eval_data_large_mcts25_eps50_fen2_{ndx_train_step}.pkl', 'wb') as picklefile:
                    pickle.dump((experience_buffer, times, evals_vs, evals_pis_1, evals_pis_n1), picklefile) # Save the experience so you can keep training with that data
                
            fig, ax = plt.subplots() 
            # ax.grid(True, linestyle=':')
            for ndx_board in range(len(eval_boards)):
                ax.plot(times, evals_vs[:, ndx_board], marker='.', linestyle='-', color=v_colors[ndx_board])
                ax.plot(times, [expected_evals[ndx_board] for _ in times], marker='', linestyle=':', color=v_colors[ndx_board])
            ax.set_xlabel('Self Play Games')
            ax.set_ylabel('Evaluation')
            ax.set_ylim((-1.1, 1.1))
            fig.set_dpi(500)
            plt.savefig('/home/mapu3213/gitrepos/DMU-_FinalProject_AlphaZero/Figs/evals_large.png')
            
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
            plt.savefig('/home/mapu3213/gitrepos/DMU-_FinalProject_AlphaZero/Figs/white2win_large.png')
            
            fig, ax = plt.subplots() 
            # ax.grid(True, linestyle=':')
            for ndx_move in range(len(evals_pis_n1.transpose())):
                ax.plot(times, evals_pis_n1[:, ndx_move], marker='.', linestyle='-', color=pi_n1_colors[ndx_move], label=pi_n1_labels[ndx_move])
            ax.plot(times, [1 for _ in times], marker='', linestyle=':', color='dodgerblue')
            ax.set_xlabel('Self Play Games')
            ax.set_ylabel('Probability')
            ax.legend(loc='best', fancybox=True, framealpha=1, handlelength=1)
            ax.set_ylim((0, 1.1))
            fig.set_dpi(500)
            plt.savefig('/home/mapu3213/gitrepos/DMU-_FinalProject_AlphaZero/Figs/black2win_large.png') 
            
        ndx_train_step += 1
    end_time_p = time.time()
    print(end_time_p - start_time_p)
        

    # profiler.disable()
    # with open('profile_results.txt', 'w') as f:
    #         stats = pstats.Stats(profiler, stream=f)
    #         stats.sort_stats('tottime')
    #         stats.print_stats()



























