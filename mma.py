from IPython import embed
import pandas as pd
from libs.ufc_scraper import update_fight_stats, update_fighter_details
import numpy as np
import datetime
from libs.Elo import Elo
from libs.ufc_scraper import *

pd.set_option('display.max_columns', 500)

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

def get_win_method_weight(x: str) -> float:
    # Adjust this to take into account strikes landed + takedowns + grappling control
    method = x.method
    round = x.round
    score = 0.

    # KOs or Subs
    if any(m in x.method for m in ['KO', 'SUB']):
        score += 5
        if round == 1:
            score += 1.5
        elif round == 2:
            score += 1
        elif round == 3:
            score += .5

    # Decisions
    elif "DEC" in method:
        if "U-" in method: # U(nanimous)-DEC
            score += 3
        # S(plit)-DEC and M(ajority)-DEC
        else:
            score += 1

    return score

def clean_fight_hist(hist_df):
    # Clean history
    hist_df = hist_df.iloc[::-1]  # invert the data so older fights are first

    # Remove loser lines (loser lines are just mirror data of winner lines)
    winner_hist_df = hist_df[hist_df.result == "W"]

    # Remove NC/DQ/Draws
    # Don't ask, I copied it off a 0 point SO answer
    keep = ["M-DEC", "U-DEC", "S-DEC", "SUB", "KO/TKO"]
    clean_hist_df = winner_hist_df[winner_hist_df.method.apply(lambda txt:
                                                               any([word_we_want in txt for word_we_want in keep]))]

    return clean_hist_df

def add_fighters(elo_scorer, clean_hist_df):
    fighter_set = set(clean_hist_df['fighter'].unique().tolist() + clean_hist_df['opponent'].unique().tolist())
    for f in fighter_set:
        elo_scorer.add_player(f)

    return elo_scorer

def update_data():
    # Update fights
    fight_hist = "C:\\Users\\dan\\PycharmProjects\\mma\\data\\fight_hist.csv"
    fight_stats = "C:\\Users\\dan\\PycharmProjects\\mma\\data\\fighter_stats.csv"
    fight_hist_updated, fighter_stats_updated = update_fights(fight_hist, fight_stats)
    write_fights(fight_hist_updated, fighter_stats_updated, fight_hist, fight_stats)

    # Parse data
    fighter_df = pd.read_csv(fight_stats)
    hist_df = pd.read_csv(fight_hist)

    return fighter_df, hist_df

def main():

    # Create Elo scorer
    elo_scorer = Elo(k=80)  # k = uncertainty, in chess 10 is used at high level, 40 at teen level

    # Update data
    fighter_df, hist_df = update_data()

    # Clean the fight history of draws/NCs and reverse it so oldest fights are last
    clean_hist_df = clean_fight_hist(hist_df)

    # Add fighters to ELO players
    elo_scorer = add_fighters(elo_scorer, clean_hist_df)

    # Set up winner/loser elo dataframes
    elo_win = np.zeros((len(clean_hist_df, )))
    elo_los = np.zeros((len(clean_hist_df, )))

    # Run calc
    monthly_top10 = {}
    prev_date = datetime.datetime.strptime("October 1, 1993", "%B %d, %Y")
    for idx, row in enumerate(clean_hist_df.itertuples()):

        # Monthly top 10
        cur_date = datetime.datetime.strptime(row.date, "%B %d, %Y")
        if prev_date < cur_date:
            sorteddict = {k: v for k, v in sorted(elo_scorer.rating_dict.items(), key=lambda item: item[1])}
            top10 = list(sorteddict.items())[-10:]
            monthly_top10[str(prev_date.month) + '/' + str(prev_date.year)] = top10
            prev_date = cur_date

        winner = row.fighter
        loser = row.opponent

        score = get_win_method_weight(row)
        elo_win[idx] = elo_scorer[winner]
        elo_los[idx] = elo_scorer[loser]
        elo_scorer.update_ratings(winner, loser, score)
        clean_hist_df.loc[:, 'elo_win'] = elo_win
        clean_hist_df.loc[:, 'elo_los'] = elo_los


    #> monthly_top10["6/2010"]
    #> top10
    #> elo_scorer.rating_dict["Anderson Silva"]
    embed()

if __name__ == "__main__":
    main()