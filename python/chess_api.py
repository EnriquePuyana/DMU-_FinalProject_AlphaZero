import chess
import numpy as np
import sys
import MCTS
import copy
import util
import concurrent.futures

# Define the piece types (pawn, knight, bishop, rook, queen, king)
# # piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
# piece_types = [chess.ROOK, chess.QUEEN, chess.KING]
# piece_letters = {chess.BISHOP: 'B', chess.ROOK: 'R', chess.QUEEN: 'Q', chess.KING: 'K'}
# move_directions = ['-N', '-S', '-E', '-W', 'NE', 'SE', 'NW', 'SW']
# action_space = np.empty((0, len(piece_types)), dtype='<U4')
# for move_direction in move_directions:
#     for distance in range(1,7+1):
#         temp_moves = np.empty((len(piece_types), 1), dtype='<U4')
#         for (ndx_piece, piece) in enumerate(piece_types):
#             temp_moves[ndx_piece] = piece_letters[piece] + move_direction + str(distance)
#         action_space = np.vstack((action_space, temp_moves.transpose()))
# action_space_size = action_space.shape
# num_actions_in_action_space = len(action_space.flatten())

# Define a function to convert a piece's position into a binary array

def can_to_tuple(move):
    # Create a mapping of files (letters) to column indices (0-7)
    file_to_col = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

    # Extract the starting and ending squares from the CAN string
    start_square = move[:2]
    end_square = move[2:4]

    # Convert the starting and ending squares to zero-based numerical coordinates
    start_row = 8 - int(start_square[1])
    start_col = file_to_col[start_square[0]]
    end_row = 8 - int(end_square[1])
    end_col = file_to_col[end_square[0]]

    # Calculate the zero-based starting square number (0-63)
    start_square_number = (start_row * 8) + start_col

    return start_square_number, (end_row, end_col)

def tuple_to_can(move_tuple):
    # Create a mapping of column indices (0-7) to files (letters)
    col_to_file = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}

    # Extract the starting square number and ending square coordinates from the tuple
    start_square_number, (end_row, end_col) = move_tuple

    # Calculate the starting square coordinates (zero-based)def to_bin
    start_row = start_square_number // 8
    start_col = start_square_number % 8

    # Convert the coordinates to file and rank representation
    start_square = col_to_file[start_col] + str(8 - start_row)
    end_square = col_to_file[end_col] + str(8 - end_row)

    # Combine the squares to form the CAN string
    can_string = start_square + end_square

    return can_string


class BoardWrapper(chess.Board):
    
    piece_types = [chess.ROOK, chess.QUEEN, chess.KING]
    piece_letters = {chess.BISHOP: 'B', chess.ROOK: 'R', chess.QUEEN: 'Q', chess.KING: 'K'}
    move_directions = ['-N', '-S', '-E', '-W', 'NE', 'SE', 'NW', 'SW']
    action_space = np.empty((0, len(piece_types)), dtype='<U4')
    for move_direction in move_directions:
        for distance in range(1,7+1):
            temp_moves = np.empty((len(piece_types), 1), dtype='<U4')
            for (ndx_piece, piece) in enumerate(piece_types):
                temp_moves[ndx_piece] = piece_letters[piece] + move_direction + str(distance)
            action_space = np.vstack((action_space, temp_moves.transpose()))
    action_space_size = action_space.shape
    num_actions_in_action_space = len(action_space.flatten())
    
    def action_2_action_space_ndx(self, pdd_move):
        return tuple([int(ndx) for ndx in np.where(self.action_space==pdd_move)])
        
    def get_turn(self):
        return (self.turn==chess.BLACK)*-2 + 1
    
    def legal_moves_func(self): 
        return self.legal_moves
    
    def piece_to_binary(self, piece_type):
        binary_array = np.zeros((8, 8), dtype=int)
        for square in chess.SQUARES:
            if self.piece_at(square) and self.piece_at(square).piece_type == piece_type:
                rank, file = chess.square_rank(square), chess.square_file(square)
                binary_array[rank][file] = int(self.piece_at(square).color)*2-1
        return binary_array
    
    def to_binary(self):
        # Create a 3D NumPy array to store the binary arrays for each piece
        binary_board = np.zeros((8, 8, len(self.piece_types)), dtype=int)
        # binary_board = []

        # Populate the chess_pieces array with binary representations of each piece
        for i, piece_type in enumerate(self.piece_types):
            binary_board[:,:,i] = self.piece_to_binary(piece_type)
            # binary_board.append(piece_to_binary(board, piece_type))
            
        return np.array(binary_board)
    
    def terminal_value(self):
        if not self.is_terminal():
            return None
        if self.is_checkmate():
            if self.turn == chess.WHITE:
                return -1
            else:
                return 1
        # if self.is_stalemate() or self.is_insufficient_material():
        #     return 0
        return 0
    
    def is_terminal(self):
        return not self.outcome(claim_draw=True)==None
    
    def nn_input_data(self):
        binary_board = self.to_binary()
        if self.turn==chess.BLACK:
            binary_board[::-1,:,:]*-1
        # return tuple([binary_board[np.newaxis, :, :], np.array(int(self.get_turn()), ndmin=2)])
        return binary_board[np.newaxis, :, :]
    
    def binary_valid_moves(self):
        binary_valid_moves_array = np.zeros(self.action_space_size, dtype=int)
        for move in self.legal_moves_func():
            binary_valid_moves_array[self.action_2_action_space_ndx(move)] = 1
            
        return binary_valid_moves_array
    
    def map_move_probs(self, move_probs):
        prob_valid_moves_array = np.zeros(self.action_space_size, dtype=float)
        legal_moves = [move for move in self.legal_moves_func()]
        if not len(legal_moves)==len(move_probs):
            print("ERROR")
            sys.exit()
        for (ndx_move, move) in enumerate(self.legal_moves_func()):
            prob_valid_moves_array[self.action_2_action_space_ndx(move)] = move_probs[ndx_move]
            
        return prob_valid_moves_array
    
    def tuple_valid_moves(self):
        return [can_to_tuple(str(move)) for move in self.legal_moves_func()]
    
    def can_valid_moves(self):
        return [str(move) for move in self.legal_moves_func()]
    
    def list_valid_moves(self):
        return [move for move in self.legal_moves_func()]
    
    # def flip_to_white(self):
    #     # Get the FEN representation of the board
    #     fen = self.fen()

    #     # Reverse the ranks (rows) in the FEN string
    #     reversed_fen_start = fen.split(' ')[0][::-1].swapcase() 
    #     reversed_fen_info = fen[len(fen.split(' ')[0]):]

    #     # Switch the case of the letters representing the pieces in the FEN string
    #     switched_colors_fen = reversed_fen_start
    #     for letter in reversed_fen_info:
                
    #         if letter=='w':
    #             letter = 'b'
    #         elif letter=='b':
    #             letter = 'w'
                
    #         switched_colors_fen += letter
        

    #     # Create a new board with the flipped and color-switched FEN
    #     flipped_and_switched_board = BoardWrapper(switched_colors_fen)

    #     return flipped_and_switched_board
    
    def move_2_pdd(self, move):
        # Get the squares from and to in the move
        from_square = move.from_square
        to_square = move.to_square

        # Calculate the distance
        to_square_file = chess.square_file(to_square)
        from_square_file = chess.square_file(from_square)
        to_square_rank = chess.square_rank(to_square)
        from_square_rank = chess.square_rank(from_square)
        rank_diff = abs(to_square_rank - from_square_rank)
        file_diff = abs(to_square_file - from_square_file)
        distance = max(rank_diff, file_diff)

        # Determine the direction
        if distance == 0:
            direction = None
        elif rank_diff == 0:
            direction = '-W' if to_square_file < from_square_file else '-E'
        elif file_diff == 0:
            direction = '-N' if to_square_rank < from_square_rank else '-S'
        elif to_square_rank<from_square_rank and to_square_file<from_square_file:
            direction = 'NW'
        elif to_square_rank<from_square_rank and to_square_file>from_square_file:
            direction = 'NE'
        elif to_square_rank>from_square_rank and to_square_file<from_square_file:
            direction = 'SW'
        elif to_square_rank>from_square_rank and to_square_file>from_square_file:
            direction = 'SE'

        # Determine the piece type
        piece_type = self.piece_letters[self.piece_type_at(from_square)]

        return piece_type + direction + str(distance)
    
    def pdd_valid_moves(self):
        return [self.move_2_pdd(move) for move in self.legal_moves_func()]
    
    def pdd_2_move(self, pdd_move):
        pdd_moves = self.pdd_valid_moves()
        if pdd_move in pdd_moves:
            return self.list_valid_moves()[pdd_moves.index(pdd_move)]
        return None
        
    def action_2_action_space_ndx(self, move):
        return tuple([int(ndx) for ndx in np.where(self.action_space==self.move_2_pdd(move))])
    
    def plan_act_remember(self, args, mcts, moves=0):
        move, move_probs = mcts.get_best_action(self)
        binary_board = self.to_binary()
        move_prob_map = self.map_move_probs(move_probs)
        if self.turn==chess.BLACK:
            binary_board = binary_board[::-1,:,:]*-1
            move_prob_map = util.flip_action_array(move_prob_map)
        turn = self.get_turn()
        if moves>args['tempThreshold']:
            self.push(move)
        else:
            self.push(self.list_valid_moves()[np.random.choice(len(move_probs), 1, p=move_probs)[0]])
        return turn, binary_board, move_prob_map
    
    def fictitious_play(self, args, NN, game_num=-1, train_step_num=-1):
        moves = 0
        binary_boards = []
        turns = []
        move_prob_maps = []
        mcts_eng = MCTS.MCTSEngine(NN, args)
        while self.terminal_value()==None:
            if moves%5 == 0:
                print(f"Move: {moves+1} in Game: {game_num+1}/{args['numEps']} of Train Step: {train_step_num+1}/{args['numTrainingEps']}")
                print(self)
                print()
            turn, binary_board, move_prob_map = self.plan_act_remember(args, mcts_eng, moves)
            binary_boards.append((binary_board))
            turns.append(turn)
            move_prob_maps.append(move_prob_map)
            moves += 1
            
        return binary_boards, turns, move_prob_maps, self.terminal_value(), self.outcome(claim_draw=True)
    
    def fictitious_epoch(self, args, NN, train_step_num=-1):
        reasons = []
        for ep in range(args['numEps']):
            examples = []
            board = copy.deepcopy(self)
            # print(f"\n\nGame {ep+1}/{args['numEps']}")
            binary_boards, turns, move_prob_maps, outcome, reason = board.fictitious_play(args, NN, game_num=ep, train_step_num=train_step_num)
            vs = np.asarray([outcome]*len(turns))
            for (bb, t, mpm, v) in zip(binary_boards, turns, move_prob_maps, vs):
                examples.append((bb, t, mpm, t*v))
            reasons.append(reason)  
            print(reason)
                
        return examples, reasons
    
    def fictitious_episode(self, args, NN, ep=-1, train_step_num=-1):
        examples = []
        # print(f"\n\nGame {ep+1}/{args['numEps']}")
        binary_boards, turns, move_prob_maps, outcome, reason = self.fictitious_play(args, NN, game_num=ep, train_step_num=train_step_num)
        vs = np.asarray([outcome]*len(turns))
        for (bb, t, mpm, v) in zip(binary_boards, turns, move_prob_maps, vs):
            examples.append((bb, mpm, t*v))
        reason
        print(reason)
                
        return examples, reason
    
    def fictitious_epoch_parallel(self, args, NN, train_step_num=-1):
        reasons = []
        exampless = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args['maxProcesses']) as executor:
            futures = [executor.submit(BoardWrapper(self.shredder_fen()).fictitious_episode, args, NN.copy(), ep=ep, train_step_num=train_step_num) for ep in range(args['numEps'])]

            for future in concurrent.futures.as_completed(futures):
                examples, reason = future.result()
                exampless.extend(examples)
                reasons.append(reason)

        return exampless, reasons
        
    
class BoardWrapper_3x3(BoardWrapper): 
    
    piece_types = [chess.ROOK, chess.QUEEN, chess.KING]
    piece_letters = {chess.BISHOP: 'B', chess.ROOK: 'R', chess.QUEEN: 'Q', chess.KING: 'K'}
    move_directions = ['-N', '-S', '-E', '-W', 'NE', 'SE', 'NW', 'SW']
    action_space = np.empty((0, len(piece_types)), dtype='<U4')
    for move_direction in move_directions:
        for distance in range(1,2+1):
            temp_moves = np.empty((len(piece_types), 1), dtype='<U4')
            for (ndx_piece, piece) in enumerate(piece_types):
                temp_moves[ndx_piece] = piece_letters[piece] + move_direction + str(distance)
            action_space = np.vstack((action_space, temp_moves.transpose()))
    action_space_size = action_space.shape
    num_actions_in_action_space = len(action_space.flatten())
    legal_3x3_squares = ['a6','a7','a8','b6','b7','b8','c6','c7','c8','8n','8b','8q','8r']
    
    def piece_to_binary(board, piece_type):
        binary_array = np.zeros((3, 3), dtype=int)
        for square in [40, 41, 42, 48, 49, 50, 56, 57, 58]:
            if board.piece_at(square) and board.piece_at(square).piece_type == piece_type:
                rank, file = chess.square_rank(square), chess.square_file(square)
                binary_array[rank-5][file] = int(board.piece_at(square).color)*2-1
        return binary_array
    
    def to_binary(self):
        # Create a 3D NumPy array to store the binary arrays for each piece
        binary_board = np.zeros((3, 3, len(self.piece_types)), dtype=int)

        # Populate the chess_pieces array with binary representations of each piece
        for i, piece_type in enumerate(self.piece_types):
            binary_board[:,:,i] = self.piece_to_binary(piece_type)
            
        return np.array(binary_board)
    
    def legal_moves_func(self, legal_3x3_squares=legal_3x3_squares): # Sorts moves just as sort_legal_moves but also removes moves that break rules for 3x3 board
        moves = []
        for move in self.legal_moves:
            if str(move)[-2:] in legal_3x3_squares: # If move is 3x3 legal
                moves.append(move)
        return moves
    
    def is_checkmate(self):
        if self.is_check():
            if len(self.legal_moves_func())>0:
                    return False
            return True
        return False
        
    def is_game_over(self):
        if self.is_checkmate():
            return True
        return len(self.legal_moves_func())==0

    def terminal_value(self):
        if not (self.is_terminal() or self.is_game_over()):
            return None
        if self.is_checkmate():
            if self.turn == chess.WHITE:
                return -1
            else:
                return 1
        return 0
    
    def fictitious_epoch_parallel(self, args, NN, train_step_num=-1):
        reasons = []
        exampless = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args['maxProcesses']) as executor:
            futures = [executor.submit(BoardWrapper_3x3(self.shredder_fen()).fictitious_episode, args, NN.copy(), ep=ep, train_step_num=train_step_num) for ep in range(args['numEps'])]

            for future in concurrent.futures.as_completed(futures):
                examples, reason = future.result()
                exampless.extend(examples)
                reasons.append(reason)

        return exampless, reasons
    