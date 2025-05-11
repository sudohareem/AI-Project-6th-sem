import pygame
import random
import math
import numpy as np
from copy import deepcopy
import time
from collections import deque

# Constants
WIDTH, HEIGHT = 1000, 800
RADIUS = 30
ROWS, COLS = 10, 10  # Larger board for Hex connections
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 150, 255)
RED = (255, 60, 60)
GREEN = (34, 139, 34)
YELLOW = (255, 215, 0)
PURPLE = (128, 0, 128)
GRAY = (100, 100, 100)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
DARK_GRAY = (50, 50, 50)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Quantum Hexxagon")
font = pygame.font.SysFont('Arial', 20)
large_font = pygame.font.SysFont('Arial', 36)
title_font = pygame.font.SysFont('Arial', 48)

class QuantumPiece:
    def __init__(self, player, probability=1.0):
        self.player = player  # 1, 2, or 3
        self.probability = probability
        self.is_king = False

class Game:
    def __init__(self, num_humans=1, ai_players=1, ai_difficulty=2):
        self.num_humans = num_humans
        self.ai_players = ai_players
        self.ai_difficulty = ai_difficulty
        self.reset_game()
        
    def reset_game(self):
        self.board = {}
        self.player_turn = 1
        self.selected = None
        self.game_over = False
        self.winner = None
        self.split_counts = {1: 0, 2: 0, 3: 0}
        self.max_splits = 3
        self.initialize_pieces()
        
        # Initialize AI players
        self.ai_instances = []
        if self.ai_players >= 1:
            self.ai_instances.append(AI(2, depth=self.ai_difficulty))
        if self.ai_players >= 2:
            self.ai_instances.append(AI(3, depth=self.ai_difficulty))
        
    def initialize_pieces(self):
        # Player 1 (Red) - Top edge (always human)
        for col in range(2, COLS-2, 2):
            self.board[(1, col)] = QuantumPiece(1)
        
        # Player 2 (Green) - Bottom edge (could be human or AI)
        for col in range(2, COLS-2, 2):
            self.board[(ROWS-2, col)] = QuantumPiece(2)
            
        # Player 3 (Blue) - Right edge (only if 3 players)
        if self.num_humans + self.ai_players == 3:
            for row in range(2, ROWS-2, 2):
                self.board[(row, COLS-2)] = QuantumPiece(3)

    def hex_to_pixel(self, row, col):
        x = RADIUS * 3 / 2 * col + 150
        y = RADIUS * math.sqrt(3) * (row + 0.5 * (col % 2)) + 100
        return int(x), int(y)

    def draw_board(self):
        screen.fill((40, 40, 40))
        
        # Draw turn indicator
        colors = {1: RED, 2: GREEN, 3: BLUE}
        turn_text = large_font.render(f"Player {self.player_turn}'s Turn", True, colors[self.player_turn])
        screen.blit(turn_text, (WIDTH - 250, 30))
        
        # Draw split counts
        split_text = font.render(f"Splits Left: {self.max_splits - self.split_counts[self.player_turn]}", True, WHITE)
        screen.blit(split_text, (WIDTH - 250, 80))
        
        # Draw exit button
        pygame.draw.rect(screen, RED, (20, 20, 100, 40))
        exit_text = font.render("Exit", True, WHITE)
        screen.blit(exit_text, (50 - exit_text.get_width()//2, 30))
        
        # Draw hex grid
        for row in range(ROWS):
            for col in range(COLS):
                x, y = self.hex_to_pixel(row, col)
                color = WHITE if (row + col) % 2 == 0 else BLACK
                self.draw_hex(x, y, RADIUS, color)
                
                # Highlight selected piece
                if self.selected and (row, col) == self.selected:
                    self.draw_hex(x, y, RADIUS-5, YELLOW)
                
                # Draw pieces
                if (row, col) in self.board:
                    piece = self.board[(row, col)]
                    piece_color = RED if piece.player == 1 else GREEN if piece.player == 2 else BLUE
                    if piece.probability < 1.0:
                        # Quantum superposition - blend colors
                        piece_color = (
                            int(piece_color[0] * piece.probability + BLUE[0] * (1-piece.probability)),
                            int(piece_color[1] * piece.probability + BLUE[1] * (1-piece.probability)),
                            int(piece_color[2] * piece.probability + BLUE[2] * (1-piece.probability))
                        )
                    pygame.draw.circle(screen, piece_color, (x, y), 15)
                    
                    # Draw probability indicator
                    if piece.probability < 1.0:
                        prob_text = font.render(f"{int(piece.probability*100)}%", True, WHITE)
                        screen.blit(prob_text, (x-15, y-10))
                    
                    # Draw king crown
                    if piece.is_king:
                        king_text = font.render("K", True, BLACK)
                        screen.blit(king_text, (x-5, y-8))

    def draw_hex(self, x, y, radius, color):
        points = []
        for i in range(6):
            angle = math.pi / 3 * i
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            points.append((px, py))
        pygame.draw.polygon(screen, color, points, 0)
        pygame.draw.polygon(screen, GRAY, points, 2)

    def get_clicked_hex(self, pos):
        mx, my = pos
        for row in range(ROWS):
            for col in range(COLS):
                x, y = self.hex_to_pixel(row, col)
                if math.hypot(mx - x, my - y) < RADIUS:
                    return (row, col)
        return None

    def get_valid_moves(self, position):
        valid_moves = []
        row, col = position
        
        if position not in self.board or self.board[position].player != self.player_turn:
            return valid_moves
        
        piece = self.board[position]
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]  # Hex directions
        
        # Regular moves (1 step)
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < ROWS and 0 <= new_col < COLS:
                if (new_row, new_col) not in self.board:
                    valid_moves.append((new_row, new_col))
        
        # Jump captures (2 steps)
        for dr, dc in directions:
            new_row, new_col = row + 2*dr, col + 2*dc
            mid_row, mid_col = row + dr, col + dc
            if 0 <= new_row < ROWS and 0 <= new_col < COLS:
                if (mid_row, mid_col) in self.board and self.board[(mid_row, mid_col)].player != self.player_turn:
                    if (new_row, new_col) not in self.board:
                        valid_moves.append((new_row, new_col))
        
        # Splitting moves (quantum mechanic)
        if self.split_counts[self.player_turn] < self.max_splits:
            for dr, dc in directions:
                new_row, new_col = row + 2*dr, col + 2*dc
                if 0 <= new_row < ROWS and 0 <= new_col < COLS:
                    if (new_row, new_col) not in self.board:
                        valid_moves.append((new_row, new_col, 'split'))  # Mark as split move
        
        # King moves (can move multiple steps in one direction)
        if piece.is_king:
            for dr, dc in directions:
                for steps in range(2, 4):  # Kings can move 2-3 steps
                    new_row, new_col = row + dr*steps, col + dc*steps
                    if 0 <= new_row < ROWS and 0 <= new_col < COLS:
                        if (new_row, new_col) not in self.board:
                            valid_moves.append((new_row, new_col))
                        else:
                            break  # Can't jump over pieces in king move
        
        return valid_moves

    def move_piece(self, start, end, is_split=False):
        if start not in self.board:
            return False
        
        piece = self.board[start]
        
        # Handle quantum probability
        if random.random() > piece.probability:
            del self.board[start]  # Piece disappears
            return False
        
        if is_split:
            if self.split_counts[self.player_turn] >= self.max_splits:
                return False
                
            # Create new piece with reduced probability
            new_prob = piece.probability * 0.5
            self.board[end] = QuantumPiece(self.player_turn, new_prob)
            piece.probability = new_prob
            self.split_counts[self.player_turn] += 1
        else:
            # Regular move or jump
            row_diff = abs(end[0] - start[0])
            col_diff = abs(end[1] - start[1])
            
            # Check if this is a jump capture (2 steps away)
            if (row_diff == 2 or col_diff == 2) and not piece.is_king:
                mid_row = (start[0] + end[0]) // 2
                mid_col = (start[1] + end[1]) // 2
                if (mid_row, mid_col) in self.board and self.board[(mid_row, mid_col)].player != self.player_turn:
                    del self.board[(mid_row, mid_col)]  # Capture
            
            # King captures (can capture by landing adjacent to opponent)
            if piece.is_king and (row_diff == 1 or col_diff == 1):
                for dr, dc in [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]:
                    adj_row, adj_col = end[0] + dr, end[1] + dc
                    if (adj_row, adj_col) in self.board and self.board[(adj_row, adj_col)].player != self.player_turn:
                        del self.board[(adj_row, adj_col)]  # King captures adjacent piece
            
            # Move the piece
            self.board[end] = piece
            del self.board[start]
            
            # Check for king promotion
            if (self.player_turn == 1 and end[0] == ROWS-1) or \
               (self.player_turn == 2 and end[0] == 0) or \
               (self.player_turn == 3 and end[1] == 0):
                piece.is_king = True
        
        return True

    def check_connection_win(self, player):
        """Check if player has connected their two sides"""
        if player == 1:  # Top to bottom
            top_edge = [(0, col) for col in range(COLS)]
            bottom_edge = [(ROWS-1, col) for col in range(COLS)]
            return self.has_path(player, top_edge, bottom_edge)
        elif player == 2:  # Left to right
            left_edge = [(row, 0) for row in range(ROWS)]
            right_edge = [(row, COLS-1) for row in range(ROWS)]
            return self.has_path(player, left_edge, right_edge)
        elif player == 3:  # Top-left to bottom-right
            top_left = [(0, 0)]
            bottom_right = [(ROWS-1, COLS-1)]
            return self.has_path(player, top_left, bottom_right)
        return False

    def has_path(self, player, start_nodes, end_nodes):
        """BFS to check if player has connected start_nodes to end_nodes"""
        visited = set()
        queue = deque()
        
        # Find all starting positions owned by player
        for node in start_nodes:
            if node in self.board and self.board[node].player == player:
                queue.append(node)
                visited.add(node)
        
        while queue:
            current = queue.popleft()
            
            # Check if we've reached any end node
            if current in end_nodes:
                return True
                
            # Explore all 6 directions
            for dr, dc in [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                if (0 <= neighbor[0] < ROWS and 0 <= neighbor[1] < COLS and 
                    neighbor in self.board and self.board[neighbor].player == player and 
                    neighbor not in visited):
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False

    def check_win_conditions(self):
        # Check piece elimination
        players_left = set(piece.player for piece in self.board.values())
        if len(players_left) == 1:
            self.game_over = True
            self.winner = players_left.pop()
            return
        
        # Check hex connection win for each player
        for player in [1, 2, 3]:
            if player not in players_left:
                continue
                
            if self.check_connection_win(player):
                self.game_over = True
                self.winner = player
                return

    def switch_player(self):
        original_player = self.player_turn
        total_players = self.num_humans + self.ai_players
        
        if total_players == 2:
            self.player_turn = 3 - self.player_turn  # Switch between 1 and 2
        else:
            self.player_turn = self.player_turn % 3 + 1  # Cycle through 1, 2, 3
        
        # Check if the next player has any valid moves
        attempts = 0
        max_attempts = total_players
        
        while attempts < max_attempts:
            # Check if the current player has any pieces and valid moves
            has_pieces = any(piece.player == self.player_turn for piece in self.board.values())
            if not has_pieces:
                # Player has no pieces left, skip to next player
                if total_players == 2:
                    self.player_turn = 3 - self.player_turn
                else:
                    self.player_turn = self.player_turn % 3 + 1
                attempts += 1
                continue
                
            # Check if player has any valid moves
            has_moves = False
            for pos in self.board:
                if self.board[pos].player == self.player_turn and self.get_valid_moves(pos):
                    has_moves = True
                    break
                    
            if has_moves:
                break  # Found a player with valid moves
            else:
                # Player has pieces but no valid moves, skip to next player
                if total_players == 2:
                    self.player_turn = 3 - self.player_turn
                else:
                    self.player_turn = self.player_turn % 3 + 1
                attempts += 1
        
        # If we've gone through all players and none can move, game is over
        if attempts >= max_attempts:
            self.game_over = True
            # Determine winner by piece count
            piece_counts = {1: 0, 2: 0, 3: 0}
            for piece in self.board.values():
                piece_counts[piece.player] += 1
            self.winner = max(piece_counts, key=piece_counts.get)

    def draw_game_over(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        
        colors = {1: RED, 2: GREEN, 3: BLUE}
        text = large_font.render(f"Player {self.winner} Wins!", True, colors[self.winner])
        restart_text = font.render("Press R to restart", True, WHITE)
        menu_text = font.render("Press M for main menu", True, WHITE)
        
        screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - 50))
        screen.blit(restart_text, (WIDTH//2 - restart_text.get_width()//2, HEIGHT//2 + 20))
        screen.blit(menu_text, (WIDTH//2 - menu_text.get_width()//2, HEIGHT//2 + 50))

    def is_ai_turn(self):
        if self.player_turn == 1:
            return False  # Player 1 is always human
        elif self.player_turn == 2 and self.ai_players >= 1 and self.num_humans < 2:
            return True
        elif self.player_turn == 3 and self.ai_players >= 2:
            return True
        return False

class AI:
    def __init__(self, player_num, depth=2):
        self.player_num = player_num
        self.depth = depth
        
    def evaluate_board(self, game):
        score = 0
        
        # Piece count advantage
        piece_counts = {1: 0, 2: 0, 3: 0}
        for piece in game.board.values():
            piece_counts[piece.player] += piece.probability  # Weight by probability
            
        # Score based on piece advantage (more weight to our pieces)
        score += 10 * piece_counts[self.player_num]
        score -= 5 * sum(piece_counts[p] for p in piece_counts if p != self.player_num)
        
        # Board control (distance to goals)
        goals = {
            1: [(ROWS-1, col) for col in range(COLS)],  # Player 1 needs to reach bottom
            2: [(0, col) for col in range(COLS)],        # Player 2 needs to reach top
            3: [(row, 0) for row in range(ROWS)]        # Player 3 needs to reach left
        }
        
        # Calculate minimum distance to goal for each piece
        min_dist = float('inf')
        for pos in game.board:
            if game.board[pos].player == self.player_num:
                for goal in goals[self.player_num]:
                    dist = self.hex_distance(pos, goal)
                    if dist < min_dist:
                        min_dist = dist
        
        # Closer is better, with exponential reward for being very close
        if min_dist != float('inf'):
            score += 50 / (min_dist + 1)
        
        # King bonus
        king_bonus = sum(1 for piece in game.board.values() 
                        if piece.player == self.player_num and piece.is_king)
        score += 15 * king_bonus
        
        # Connection potential - check if we're close to connecting our sides
        if game.check_connection_win(self.player_num):
            score += 1000  # Immediate win
        
        # Threat detection - check if opponents are close to winning
        opponents = [p for p in [1, 2, 3] if p != self.player_num]
        for opp in opponents:
            if game.check_connection_win(opp):
                score -= 1000  # Immediate loss
        
        return score
    
    def hex_distance(self, pos1, pos2):
        """Calculate hexagonal grid distance between two positions"""
        x1, y1 = pos1
        x2, y2 = pos2
        dx = x2 - x1
        dy = y2 - y1
        return (abs(dx) + abs(dy) + abs(dx - dy)) // 2
    
    def minimax(self, game, depth, alpha, beta, maximizing_player):
        if depth == 0 or game.game_over:
            return self.evaluate_board(game), None
        
        best_move = None
        if maximizing_player:
            max_eval = float('-inf')
            for move in self.get_all_possible_moves(game, self.player_num):
                new_game = deepcopy(game)
                if len(move) == 3 and move[2] == 'split':
                    new_game.move_piece(move[0], move[1], True)
                else:
                    new_game.move_piece(move[0], move[1])
                new_game.check_win_conditions()
                new_game.switch_player()
                
                evaluation, _ = self.minimax(new_game, depth-1, alpha, beta, False)
                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = move
                
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            opponents = [p for p in [1, 2, 3] if p != self.player_num]
            for opponent in opponents:
                for move in self.get_all_possible_moves(game, opponent):
                    new_game = deepcopy(game)
                    if len(move) == 3 and move[2] == 'split':
                        new_game.move_piece(move[0], move[1], True)
                    else:
                        new_game.move_piece(move[0], move[1])
                    new_game.check_win_conditions()
                    new_game.switch_player()
                    
                    evaluation, _ = self.minimax(new_game, depth-1, alpha, beta, True)
                    if evaluation < min_eval:
                        min_eval = evaluation
                        best_move = move
                    
                    beta = min(beta, evaluation)
                    if beta <= alpha:
                        break
            return min_eval, best_move
    
    def get_all_possible_moves(self, game, player_num):
        moves = []
        for pos in game.board:
            if game.board[pos].player == player_num:
                valid_moves = game.get_valid_moves(pos)
                for move in valid_moves:
                    if len(move) == 3 and move[2] == 'split':
                        moves.append((pos, (move[0], move[1]), 'split'))
                    else:
                        moves.append((pos, move))
        return moves
    
    def make_move(self, game):
        start_time = time.time()
        _, best_move = self.minimax(game, self.depth, float('-inf'), float('inf'), True)
        end_time = time.time()
        print(f"AI move took {end_time - start_time:.2f} seconds")
        
        if best_move:
            if len(best_move) == 3 and best_move[2] == 'split':
                game.move_piece(best_move[0], best_move[1], True)
            else:
                game.move_piece(best_move[0], best_move[1])
            game.check_win_conditions()
            if not game.game_over:
                game.switch_player()

def draw_menu():
    screen.fill(DARK_GRAY)
    
    title = title_font.render("HEXXAGON", True, CYAN)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 100))
    
    # Player options
    pygame.draw.rect(screen, BLUE, (WIDTH//2 - 150, 200, 300, 60))
    pygame.draw.rect(screen, GREEN, (WIDTH//2 - 150, 280, 300, 60))
    pygame.draw.rect(screen, ORANGE, (WIDTH//2 - 150, 360, 300, 60))
    pygame.draw.rect(screen, PURPLE, (WIDTH//2 - 150, 440, 300, 60))
    
    
    
    player1_text = large_font.render("1 Player (Human vs AI)", True, WHITE)
    player2_text = large_font.render("2 Players (Human vs Human)", True, WHITE)
    player1ai_text = large_font.render("1 Human + 1 AI", True, WHITE)
    player2ai_text = large_font.render("2 Humans + 1 AI", True, WHITE)
    
    
    
    screen.blit(player1_text, (WIDTH//2 - player1_text.get_width()//2, 215))
    screen.blit(player2_text, (WIDTH//2 - player2_text.get_width()//2, 295))
    screen.blit(player1ai_text, (WIDTH//2 - player1ai_text.get_width()//2, 375))
    screen.blit(player2ai_text, (WIDTH//2 - player2ai_text.get_width()//2, 455))
    
    
    # Difficulty options
    pygame.draw.rect(screen, RED, (WIDTH//2 - 150, 520, 300, 60))
    pygame.draw.rect(screen, ORANGE, (WIDTH//2 - 150, 600, 300, 60))
    pygame.draw.rect(screen, YELLOW, (WIDTH//2 - 150, 680, 300, 60))
    
    easy_text = large_font.render("Easy", True, WHITE)
    medium_text = large_font.render("Medium", True, WHITE)
    hard_text = large_font.render("Hard", True, WHITE)
    
    screen.blit(easy_text, (WIDTH//2 - easy_text.get_width()//2, 535))
    screen.blit(medium_text, (WIDTH//2 - medium_text.get_width()//2, 615))
    screen.blit(hard_text, (WIDTH//2 - hard_text.get_width()//2, 695))
    
    # Exit button
    pygame.draw.rect(screen, RED, (WIDTH//2 - 150, 760, 300, 60))
    exit_text = large_font.render("Exit Game", True, WHITE)
    screen.blit(exit_text, (WIDTH//2 - exit_text.get_width()//2, 775))
    
    pygame.display.flip()

def main():
    clock = pygame.time.Clock()
    running = True
    in_menu = True
    num_humans = 1
    ai_players = 1
    ai_difficulty = 2  # Default medium difficulty
    game = None
    
    while running:
        if in_menu:
            draw_menu()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    
                    # Check if exit button was clicked
                    if WIDTH//2 - 150 <= mx <= WIDTH//2 + 150 and 760 <= my <= 820:
                        running = False
                    
                    # Check player selection
                    if WIDTH//2 - 150 <= mx <= WIDTH//2 + 150:
                        if 200 <= my <= 260:  # 1 Player (Human vs AI)
                            num_humans = 1
                            ai_players = 1
                            in_menu = False
                            game = Game(num_humans=num_humans, ai_players=ai_players, ai_difficulty=ai_difficulty)
                        elif 280 <= my <= 340:  # 2 Players (Human vs Human)
                            num_humans = 2
                            ai_players = 0
                            in_menu = False
                            game = Game(num_humans=num_humans, ai_players=ai_players, ai_difficulty=ai_difficulty)
                        elif 360 <= my <= 420:  # 1 Human + 1 AI
                            num_humans = 1
                            ai_players = 1
                            in_menu = False
                            game = Game(num_humans=num_humans, ai_players=ai_players, ai_difficulty=ai_difficulty)
                        elif 440 <= my <= 500:  # 2 Humans + 1 AI
                            num_humans = 2
                            ai_players = 1
                            in_menu = False
                            game = Game(num_humans=num_humans, ai_players=ai_players, ai_difficulty=ai_difficulty)
                        
                        # Check difficulty selection
                        if 520 <= my <= 580:  # Easy
                            ai_difficulty = 1
                        elif 600 <= my <= 660:  # Medium
                            ai_difficulty = 2
                        elif 680 <= my <= 740:  # Hard
                            ai_difficulty = 3
        else:
            # Check if exit button was clicked
            mouse_pos = pygame.mouse.get_pos()
            if 20 <= mouse_pos[0] <= 120 and 20 <= mouse_pos[1] <= 60:
                pygame.draw.rect(screen, (255, 0, 0), (20, 20, 100, 40), 2)  # Highlight exit button on hover
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        in_menu = True
            else:
                pygame.draw.rect(screen, RED, (20, 20, 100, 40))
            
            exit_text = font.render("Exit", True, WHITE)
            screen.blit(exit_text, (50 - exit_text.get_width()//2, 30))
            
            # AI move if it's an AI player's turn
            if not game.game_over and game.is_ai_turn():
                ai_index = game.player_turn - 2  # AI players are 2 and 3 (index 0 and 1)
                game.ai_instances[ai_index].make_move(game)
                pygame.time.delay(500)  # Add slight delay so AI move is visible
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and game.game_over:
                        game.reset_game()
                    if event.key == pygame.K_m and game.game_over:
                        in_menu = True
                    if event.key == pygame.K_ESCAPE:
                        in_menu = True
                
                if not game.game_over and not game.is_ai_turn() and event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if exit button was clicked
                    if 20 <= event.pos[0] <= 120 and 20 <= event.pos[1] <= 60:
                        in_menu = True
                        continue
                    
                    pos = game.get_clicked_hex(event.pos)
                    if pos:
                        if game.selected is None:
                            # Select a piece belonging to current player
                            if pos in game.board and game.board[pos].player == game.player_turn:
                                game.selected = pos
                        else:
                            valid_moves = game.get_valid_moves(game.selected)
                            move_made = False
                            
                            for move in valid_moves:
                                if len(move) == 3 and move[2] == 'split' and pos == (move[0], move[1]):
                                    if game.move_piece(game.selected, pos, True):
                                        move_made = True
                                        break
                                elif pos == move:
                                    if game.move_piece(game.selected, pos):
                                        move_made = True
                                        break
                            
                            if move_made:
                                game.check_win_conditions()
                                if not game.game_over:
                                    game.switch_player()
                            game.selected = None
            
            # Drawing
            game.draw_board()
            
            # Highlight valid moves
            if game.selected:
                valid_moves = game.get_valid_moves(game.selected)
                for move in valid_moves:
                    x, y = game.hex_to_pixel(move[0], move[1])
                    if len(move) == 3 and move[2] == 'split':
                        pygame.draw.circle(screen, ORANGE, (x, y), 10)
                    else:
                        pygame.draw.circle(screen, PURPLE, (x, y), 10)
            
            if game.game_over:
                game.draw_game_over()
            
            pygame.display.flip()
        
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()