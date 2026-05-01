import chess


PIECE_TO_ID = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

PHASE_TO_ID = {
    "opening": 0,
    "middlegame": 1,
    "endgame": 2,
}

TIME_CLASS_TO_ID = {
    "Rapid": 0,
    "Classical": 1,
}

RESULT_TO_ID = {
    "white_win": 0,
    "black_win": 1,
    "draw": 2,
    "unknown": 3,
}


class MoveTokenizer:
    def __init__(self, rating_bucket_size=100):
        self.rating_bucket_size = rating_bucket_size

    def rating_to_bucket(self, rating):
        if rating is None:
            return 0
        return max(0, int(rating) // self.rating_bucket_size)

    def infer_phase(self, board, ply_index):
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
            board.pieces(chess.QUEEN, chess.BLACK)
        )
        minor_major_pieces = sum(
            len(board.pieces(piece_type, color))
            for piece_type in (chess.ROOK, chess.BISHOP, chess.KNIGHT)
            for color in (chess.WHITE, chess.BLACK)
        )

        if ply_index < 20:
            return "opening"
        if queens == 0 or minor_major_pieces <= 4:
            return "endgame"
        return "middlegame"

    def encode_game(
        self,
        moves,
        white_rating,
        black_rating,
        time_class=None,
        result_label="unknown",
    ):
        board = chess.Board()
        encoded_moves = []

        for ply_index, move_uci in enumerate(moves):
            move = chess.Move.from_uci(move_uci)
            piece = board.piece_at(move.from_square)
            is_capture = board.is_capture(move)
            is_check = board.gives_check(move)
            moving_side = 0 if board.turn == chess.WHITE else 1
            player_rating = white_rating if board.turn == chess.WHITE else black_rating

            if piece is None:
                raise ValueError(f"Invalid move sequence. No piece on {move.from_square}.")

            phase_name = self.infer_phase(board, ply_index)

            encoded_moves.append(
                {
                    "move_uci": move_uci,
                    "from_square": move.from_square,
                    "to_square": move.to_square,
                    "promotion": move.promotion or 0,
                    "piece_type": PIECE_TO_ID[piece.piece_type],
                    "is_capture": int(is_capture),
                    "is_check": int(is_check),
                    "moving_side": moving_side,
                    "ply_index": ply_index,
                    "phase_id": PHASE_TO_ID[phase_name],
                    "phase_name": phase_name,
                    "player_rating": int(player_rating),
                    "opponent_rating": int(
                        black_rating if board.turn == chess.WHITE else white_rating
                    ),
                    "rating_bucket": self.rating_to_bucket(player_rating),
                    "time_class_id": TIME_CLASS_TO_ID.get(time_class, -1),
                    "result_id": RESULT_TO_ID.get(result_label, RESULT_TO_ID["unknown"]),
                }
            )

            board.push(move)

        return encoded_moves
