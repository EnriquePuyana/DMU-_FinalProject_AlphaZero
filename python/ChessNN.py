import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, Dense, Dropout, Reshape, Conv3D, Concatenate
from tensorflow.keras.optimizers import Adam
import chess_api
import chess
import MCTS
import copy
import time
import util

board_normal = chess_api.BoardWrapper()
board_3x3 = chess_api.BoardWrapper_3x3()

args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 5,
    'batch_size': 200,
    'shuffle': True,
    'cuda': False,
}

class ChessNN():
    def __init__(self, args, three_by_three=False):
        if three_by_three:
            self.board_x, self.board_y = 3, 3
            self.num_channels = len(board_3x3.piece_types)  # Number of channels for the input
            data_shape = (self.board_x, self.board_y, self.num_channels, 1)

            # Neural Net
            self.input_boards = Input(shape=data_shape)
            # input_channels = [Reshape((self.board_x, self.board_y, 1))(input_board) for input_board in self.input_boards]
            h_conv1 = Activation('relu')((Conv3D(100, (2, 2, self.num_channels), padding='same', use_bias=False, input_shape=data_shape))(self.input_boards))
            h_conv2 = Activation('relu')((Conv3D(100, (2, 2, self.num_channels), padding='same', use_bias=False)(h_conv1)))
            # h_conv3 = Activation('relu')((Conv3D(512, 2, padding='same', use_bias=False)(h_conv2)))
            # h_conv4 = Activation('relu')((Conv3D(512, 2, padding='same', use_bias=False)(h_conv3)))
            h_conv4_flat = Flatten()(h_conv2)

            s_fc1 = Dropout(args['dropout'])(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))
            s_fc2 = Dropout(args['dropout'])(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))

            self.pi = Dense(board_3x3.num_actions_in_action_space, activation='softmax', name='pi')(s_fc2)
            self.pi_shaped = Reshape(board_3x3.action_space_size)(self.pi)
            self.v = Dense(1, activation='tanh', name='v')(s_fc2)

            self.model = Model(inputs=[self.input_boards], outputs=[self.pi_shaped, self.v])
            self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args['lr']))
            
        else:
            # game params
            self.board_x, self.board_y = 8, 8
            # self.action_size = game.getActionSize() 
            self.num_channels = len(board_normal.piece_types)  # Number of channels for the input
            data_shape = (self.board_x, self.board_y, self.num_channels, 1)

            # Neural Net
            self.input_boards = Input(shape=data_shape)
            # input_channels = [Reshape((self.board_x, self.board_y, 1))(input_board) for input_board in self.input_boards]
            h_conv1 = Activation('relu')((Conv3D(100, 3, padding='same', use_bias=False, input_shape=data_shape))(self.input_boards))
            h_conv2 = Activation('relu')((Conv3D(100, 3, padding='same', use_bias=False)(h_conv1)))
            # h_conv3 = Activation('relu')((Conv3D(512, 2, padding='same', use_bias=False)(h_conv2)))
            # h_conv4 = Activation('relu')((Conv3D(512, 2, padding='same', use_bias=False)(h_conv3)))
            h_conv4_flat = Flatten()(h_conv2)
            
            # turn_input = Input(shape=(1,), name='turn_input')
            # conv_and_turn = Concatenate()([h_conv4_flat, turn_input])

            s_fc1 = Dropout(args['dropout'])(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))
            s_fc2 = Dropout(args['dropout'])(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))

            self.pi = Dense(board_normal.num_actions_in_action_space, activation='softmax', name='pi')(s_fc2)
            self.pi_shaped = Reshape(board_normal.action_space_size)(self.pi)
            self.v = Dense(1, activation='tanh', name='v')(s_fc2)

            self.model = Model(inputs=[self.input_boards], outputs=[self.pi_shaped, self.v])
            self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args['lr']))

class NNetWrapper(ChessNN):
    def __init__(self, three_by_three=False):
        self.three_by_three = three_by_three
        self.nnet = ChessNN(args, three_by_three=self.three_by_three)
        
    def copy(self):
        # Create a new instance of the NNetWrapper class
        copied_nnet_wrapper = NNetWrapper(three_by_three = self.three_by_three)
        # Clone the Keras model inside the new instance
        copied_nnet_wrapper.nnet.model = clone_model(self.nnet.model)
        # Copy the weights from the original model to the copied model
        copied_nnet_wrapper.nnet.model.set_weights(self.nnet.model.get_weights())

        return copied_nnet_wrapper

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        # turns = np.asarray(turns)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x = [input_boards], y = [target_pis, target_vs], batch_size = args['batch_size'], epochs = args['epochs'], shuffle=args['shuffle'])

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()
        pi, v = self.nnet.model.predict(board.nn_input_data(), verbose=False)
        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        if board.turn==chess.BLACK:
            pi[0] = util.flip_action_array(pi[0])
            v[0] = -v[0]
        return pi[0], v[0]
    
    def get_best_actions(self, board):
        """
        board: np array with board
        """
        pi, v = self.predict(board)
        valids = board.binary_valid_moves()
        pi = pi*valids  # masking invalid moves
        sum_Ps_s = np.sum(pi)
        if sum_Ps_s > 0:
            pi /= sum_Ps_s  # renormalize

        ndx_as = [board.action_2_action_space_ndx(a) for a in board.legal_moves_func()]
        action_probs = [pi[ndx_a] for ndx_a in ndx_as]

        return [(board.list_valid_moves()[ndx], action_probs[ndx]) for ndx in np.argsort(action_probs)[::-1]], v
    
    def get_valid_action_probs(self, board):
        """
        board: np array with board
        """
        pi, v = self.predict(board)
        valids = board.binary_valid_moves()
        pi = pi*valids  # masking invalid moves
        sum_Ps_s = np.sum(pi)
        if sum_Ps_s > 0:
            pi /= sum_Ps_s  # renormalize

        ndx_as = [board.action_2_action_space_ndx(a) for a in board.legal_moves_func()]
        action_probs = [pi[ndx_a] for ndx_a in ndx_as]

        return action_probs, v

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))

        self.nnet.model.load_weights(filepath)
        
        
    
