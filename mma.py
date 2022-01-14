import requests
import urllib3
from bs4 import BeautifulSoup
from IPython import embed
import pandas as pd
pd.set_option('display.max_columns', 500)
from scripts.ufc_scraper import update_fight_stats, update_fighter_details
import seaborn as sns
import matplotlib.pyplot as plt

def update(fight_hist, fight_stats):
    """
    Returns the updated fight history and fighter stats dataframes
    """
    # get old data
    fight_hist_old = pd.read_csv(fight_hist)
    fighter_stats_old = pd.read_csv(fight_stats)

    # update fight history
    fight_hist_updated = update_fight_stats(fight_hist_old)

    # update fighter stats
    fighter_stats_updated = update_fighter_details(fight_hist_updated.fighter_url.unique(), fighter_stats_old)

    return fight_hist_updated, fighter_stats_updated

def write_fights(fight_hist_updated, fighter_stats_updated, fight_hist, fight_stats):
    """
    Write dataframe to csv
    """
    fight_hist_updated.to_csv(fight_hist, index = False)
    fighter_stats_updated.to_csv(fight_stats, index = False)

def update_fights(fight_hist, fight_stats):
    fight_hist_updated, fighter_stats_updated = update(fight_hist, fight_stats)
    write_fights(fight_hist_updated, fighter_stats_updated, fight_hist, fight_stats)

    return fight_hist_updated, fighter_stats_updated

###################### ELO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


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

    def add_player(self, name: str, rating: float = 1500.):
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
        return 1 / ((10.0 ** (exp)) + 1)

def get_win_method_weight(x: str) -> float:
    # Adjust this to take into account strikes landed + takedowns + grappling control
    if "DEC" in x:
        if "U-" in x: # U(nanimous)-DEC
            return 3.
        else: # S(plit)-DEC and M(ajority)-DEC
            return 1.
    elif "KO" in x:
        return 5.
    elif "SUB" in x:
        return 5.
    else:
        return 0.


def main():
    # Update fights
    fight_hist = "C:\\Users\\dan\\PycharmProjects\\mma\\data\\fight_hist.csv"
    fight_stats = "C:\\Users\\dan\\PycharmProjects\\mma\\data\\fighter_stats.csv"
    # fight_hist_updated, fighter_stats_updated = update_fights(fight_hist, fight_stats)
    # write_fights(fight_hist_updated, fighter_stats_updated, fight_hist, fight_stats)

    # Parse data
    elo_scorer = Elo(k=24)
    fighter_df = pd.read_csv(fight_stats)
    hist_df = pd.read_csv(fight_hist)

    # Clean history
    hist_df = hist_df.iloc[::-1] # invert the data so older fights are first
    # Don't ask, I copied it off a 0 point SO answer
    keep = ["M-DEC", "U-DEC", "S-DEC", "SUB", "KO/TKO"]
    clean_hist_df = hist_df[hist_df.method.apply(lambda txt:
                                                 any([word_we_want in txt for word_we_want in keep]))]
    elo_win = np.zeros((len(clean_hist_df, )))
    elo_los = np.zeros((len(clean_hist_df, )))

    # Add fighters to ELO players
    fighter_set = set(clean_hist_df['fighter'].unique().tolist() + clean_hist_df['opponent'].unique().tolist())
    for f in fighter_set:
        elo_scorer.add_player(f)

    # Run calc
    for idx, row in enumerate(clean_hist_df.itertuples()):
        if row.result == 'W':
            winner = row.fighter
            loser = row.opponent
        else:
            winner = row.opponent
            loser = row.fighter
        score = get_win_method_weight(row.method)
        elo_win[idx] = elo_scorer[winner]
        elo_los[idx] = elo_scorer[loser]
        elo_scorer.update_ratings(winner, loser, score)
        clean_hist_df.loc[:, 'elo_win'] = elo_win
        clean_hist_df.loc[:, 'elo_los'] = elo_los
    embed()

if __name__ == "__main__":
    main()