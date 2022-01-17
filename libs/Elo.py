import numpy as np


class Elo:

    def __init__(self, k: int):
        self.rating_dict = {}
        self.k = k

    def __getitem__(self, player: str) -> float:
        return self.rating_dict[player]

    def __setitem__(self, player: str, data: float) -> None:
        self.rating_dict[player] = data

    def get_margin_factor(self, score: float) -> float:
        return np.log2(score + 1)

    def get_inflation_factor(self, r_win: float, r_lose: float) -> float:
        return 1 / (1 - ((r_lose - r_win) / 2200))

    def add_player(self, name: str, rating: float = 1200.):
        self.rating_dict[name] = rating

    def update_ratings(self, winner: str, loser: str, score: float) -> None:
        expected_result = self.get_expect_result(
            self.rating_dict[winner], self.rating_dict[loser]
        )
        margin_factor = self.get_margin_factor(score)
        inflation_factor = self.get_inflation_factor(
            self.rating_dict[winner], self.rating_dict[loser]
        )
        self.rating_dict[winner] = self.rating_dict[winner] + self.k * margin_factor * inflation_factor * (
                    1 - expected_result)
        self.rating_dict[loser] = self.rating_dict[loser] + self.k * margin_factor * inflation_factor * (
                    -1 + expected_result)

    def get_expect_result(self, p1: float, p2: float) -> float:
        exp = (p2 - p1) / 400.0
        return 1 / ((10.0 ** exp) + 1)
