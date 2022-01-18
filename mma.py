from IPython import embed
import pandas as pd
from libs.ufc_scraper import update_fight_stats, update_fighter_details
import numpy as np
import datetime
from libs.Elo import Elo
from libs.ufc_scraper import *
import matplotlib.pyplot as mp
import seaborn as sb

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

def remove_draws(hist_df):

    # Just keep the majority dec, unanimous dec, split dec, sub, and KO finishes
    keep = ["M-DEC", "U-DEC", "S-DEC", "SUB", "KO/TKO"]
    # Don't ask, I copied it off a 0 point SO answer
    finish_only_hist_df = hist_df[hist_df.method.apply(lambda txt:
                                                    any([word_we_want in txt for word_we_want in keep]))]

    # Remove NC/DQ/Draws
    no_draw_hist_df = finish_only_hist_df[finish_only_hist_df.result != "D"]

    return no_draw_hist_df

def add_fighters(elo_scorer, unmirrored_hist_df):
    fighter_set = set(unmirrored_hist_df['fighter'].unique().tolist() + unmirrored_hist_df['opponent'].unique().tolist())
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

def stat_differential(df, stat_index):
    """
    Stat indexes:
        0 date
        1 fight_url
        2 event_url
        3 result
        4 fighter
        5 opponent
        6 division
        7 method
        8 round
        9 time
        10 fighter_url
        11 opponent_url
        12 knockdowns
        13 sub_attempts
        14 pass
        15 reversals
        16 takedowns_landed
        17 takedowns_attempts
        18 sig_strikes_landed
        19 sig_strikes_attempts
        20 total_strikes_landed
        21 total_strikes_attempts
        22 head_strikes_landed
        23 head_strikes_attempts
        24 body_strikes_landed
        25 body_strikes_attempts
        26 leg_strikes_landed
        27 leg_strikes_attempts
        28 distance_strikes_landed
        29 distance_strikes_attempts
        30 clinch_strikes_landed
        31 clinch_strikes_attempts
        32 ground_strikes_landed
        33 ground_strikes_attempts
    :param df: dataframe
    :param cur_stat_index: index from the row tuple that the stat we're interested in is placed
    :return: the list which will be appended as a column to the dataframe
    """
    #There is one extra stat at the beginning of the tuple, Index=432354 when you iter over the tuple
    stat_index += 1
    prev_fight = None
    total_stat_diff = []
    # Go through each row, save the previous values then divide to get strike diff
    for row in df.itertuples():
        cur_stat = row[stat_index]
        # We can't divide by 0 so we just set an opponent with 0 stat to 1
        # How is this gonna affect low number stats like takedowns or submissions?
        if cur_stat == 0:
            cur_stat = 1
        fighter = row.fighter
        opponent = row.opponent
        if not prev_fight:
            prev_fight = {fighter: cur_stat}
            continue
        elif row.opponent in prev_fight:
            prev_diff = prev_fight[opponent] / cur_stat
            cur_diff = cur_stat / prev_fight[opponent]
            total_stat_diff.append(prev_diff)
            total_stat_diff.append(cur_diff)
            prev_fight = None
        else:
            raise Exception

    return total_stat_diff

def main():

    # Create Elo scorer
    elo_scorer = Elo(k=75)  # k = uncertainty, in chess 10 is used at high level, 40 at teen level

    # Update data
    fighter_df, hist_df = update_data()

    # Invert the data so older fights are first
    hist_df = hist_df.iloc[::-1]

    # Clean the fight history of draws/NCs
    no_draw_hist_df = remove_draws(hist_df)

    # Convert result column from object to int64
    no_draw_hist_df['result'] = no_draw_hist_df.result.str.replace('W', '1')
    no_draw_hist_df['result'] = no_draw_hist_df.result.str.replace('L', '0')
    no_draw_hist_df['result'] = no_draw_hist_df.result.astype(np.int64)

    # Add second tier variables
    # accuracy?
    # Get a dictionary of row name : index number so we can call the index number in a for loop
    new_vars_hist_df = no_draw_hist_df.copy()

    differential_stats = {}
    for idx, stat in enumerate(no_draw_hist_df):
        # 13 is where the fight stats start and reversals are the only stat that is a string
        if idx >= 12:
            if stat != 'reversals':
                differential_stats[stat] = idx

    for stat in differential_stats:                          # this is the stat index within the row tuple
        total_stat_diff = stat_differential(no_draw_hist_df, differential_stats[stat])
        new_vars_hist_df[stat+'_differential'] = total_stat_diff

    # Remove loser lines (loser lines are just mirror data of winner lines)
    unmirrored_hist_df = no_draw_hist_df[no_draw_hist_df.result == 1]

    # Add fighters to ELO players
    elo_scorer = add_fighters(elo_scorer, unmirrored_hist_df)

    # Set up winner/loser elo dataframes
    elo_win = np.zeros((len(unmirrored_hist_df, )))
    elo_los = np.zeros((len(unmirrored_hist_df, )))

    # Run calc
    monthly_top25 = {}
    prev_date = datetime.datetime.strptime("October 1, 1993", "%B %d, %Y")
    for idx, row in enumerate(unmirrored_hist_df.itertuples()):

        # Monthly top 25
        cur_date = datetime.datetime.strptime(row.date, "%B %d, %Y")
        if prev_date < cur_date:
            # Sort by top fighter elo
            sorteddict = {k: v for k, v in sorted(elo_scorer.rating_dict.items(), key=lambda item: item[1])}
            # Latest top 25
            top25 = list(sorteddict.items())[-25:]
            # Monthly top10 callable via: monthly_top10["6/2010"]
            monthly_top25[str(prev_date.month) + '/' + str(prev_date.year)] = top25
            prev_date = cur_date

        winner = row.fighter
        loser = row.opponent

        score = get_win_method_weight(row)
        elo_win[idx] = elo_scorer[winner]
        elo_los[idx] = elo_scorer[loser]
        elo_scorer.update_ratings(winner, loser, score)
        unmirrored_hist_df.loc[:, 'elo_win'] = elo_win
        unmirrored_hist_df.loc[:, 'elo_los'] = elo_los


    #> monthly_top10["6/2010"]
    #> top10
    #> elo_scorer.rating_dict["Anderson Silva"]
    embed()

if __name__ == "__main__":
    main()