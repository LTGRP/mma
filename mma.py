from IPython import embed
import pandas as pd
import numpy as np
import datetime
from libs.Elo import Elo
from libs.ufc_scraper import *
import matplotlib.pyplot as mp
import seaborn as sb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
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
    """
    Convert result column from object to int64
    :param hist_df: historical stats dataframe 
    :return: cleaned dataframe
    """

    # Just keep the majority dec, unanimous dec, split dec, sub, and KO finishes
    keep = ["M-DEC", "U-DEC", "S-DEC", "SUB", "KO/TKO"]
    # Don't ask, I copied it off a 0 point SO answer
    finish_only_hist_df = hist_df[hist_df.method.apply(lambda txt:
                                                    any([word_we_want in txt for word_we_want in keep]))]

    # Remove NC/DQ/Draws
    no_draw_hist_df = finish_only_hist_df[finish_only_hist_df.result != "D"]
    no_draw_hist_df['result'] = no_draw_hist_df.result.str.replace('W', '1')
    no_draw_hist_df['result'] = no_draw_hist_df.result.str.replace('L', '0')
    no_draw_hist_df['result'] = no_draw_hist_df.result.astype(np.int64)

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

def stat_differential(df, stat_index, differential=False, accuracy=False):
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
        # Implement acc here (attempted/landed)
        if accuracy:
            pass

        # Calc the differential
        elif differential:
            cur_stat = row[stat_index]
            fighter = row.fighter
            opponent = row.opponent
            if not prev_fight:
                prev_fight = {fighter: cur_stat}
                continue

            # Dealing with 0 in denominator
            elif row.opponent in prev_fight:

                try:
                    prev_diff = prev_fight[opponent] / cur_stat
                except ZeroDivisionError:
                    prev_diff = 0

                try:
                    cur_diff = cur_stat / prev_fight[opponent]
                except ZeroDivisionError:
                    cur_diff = 0

                total_stat_diff.append(prev_diff)
                total_stat_diff.append(cur_diff)
                prev_fight = None
            else:
                raise Exception

    return total_stat_diff

def calc_brier_score(new_vars_hist_df):
    # reversals is a string
    train_df = new_vars_hist_df.copy().drop(columns="reversals")

    # Toss all the objects and leave all the int64 stats
    for idx, col in enumerate(new_vars_hist_df):
        if idx < 12:
            train_df = train_df.drop(columns=col)

    brier_score = {}
    y = new_vars_hist_df["result"]
    for col in train_df:
        # Logistic regression
        X = train_df[[col]]  # double [[]] to made it a 2D array
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        lr = LogisticRegression(max_iter=3000)
        lr.fit(X_train, y_train)
        score = lr.score(X_test, y_test)

        # Brier score
        probs = lr.predict_proba(X_test)
        probs = probs[:, 1]  # Keeping only the values in positive label
        loss = brier_score_loss(y_test, probs)
        brier_score[col] = loss

    return brier_score

def peak_elo(historical_elo, name):
    s = historical_elo[name]
    s.sort(key=lambda x: x[1])
    peak = s[-1]
    return peak

def get_differential_stats(df):
    differential_stats = {}
    for idx, stat in enumerate(df):
        # 13 is where the fight stats start and reversals are the only stat that is a string
        if idx >= 12:
            if stat != 'reversals':
                differential_stats[stat] = idx
    return differential_stats

def update_diff_stats(new_vars_hist_df, no_draw_hist_df, differential_stats):
    for stat in differential_stats:                          # this is the stat index within the row tuple
        total_stat_diff = stat_differential(no_draw_hist_df, differential_stats[stat], differential=True)
        new_vars_hist_df[stat+'_differential'] = total_stat_diff
    return new_vars_hist_df

def run_elo_calc(unmirrored_hist_df, elo_scorer):
    historical_elo = {}
    monthly_top25 = {}
    # Set up winner/loser elo dataframes
    elo_win = np.zeros((len(unmirrored_hist_df, )))
    elo_los = np.zeros((len(unmirrored_hist_df, )))

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

        if winner in historical_elo:
            historical_elo[winner].append((loser, elo_scorer[winner]))
        else:
            historical_elo[winner] = [(loser, elo_scorer[winner])]
        if loser in historical_elo:
            historical_elo[loser].append((winner, elo_scorer[loser]))
        else:
            historical_elo[loser] = [(winner, elo_scorer[loser])]

    return elo_scorer, historical_elo, monthly_top25, top25

def create_diff_dfs(new_vars_hist_df, avg_hist_diff_fighters):
    # Add new columns to fighter df
    avg_hist_diff_fights = new_vars_hist_df.copy()
    for col in new_vars_hist_df:
        if "differential" in col:
            avg_hist_diff_fights.loc[:, "historical_avg_" + col] = 0
            avg_hist_diff_fighters.loc[:, "total_" + col] = 0

    return avg_hist_diff_fights, avg_hist_diff_fighters

def hist_diff_brier_score(df):
    b = {}
    brier_score = calc_brier_score(df)
    brier_score_eq = {k: v for k, v in sorted(brier_score.items(), key=lambda item: item[1])}
    # just history
    for k, v in brier_score_eq.items():
        if "hist" in k:
            b[k] = v
    return b


def add_hist_diff_stats(new_vars_hist_df, avg_hist_diff_fights, avg_hist_diff_fighters):
    for row in new_vars_hist_df.itertuples():
        f = row.fighter
        fighter_index = avg_hist_diff_fighters.loc[avg_hist_diff_fighters["name"] == f].index

        # Add fighter differentials coming into the fight (not including the fight)
        fighter_counted_fights = avg_hist_diff_fighters.loc[fighter_index].counted_fights.values[0]

        # Get all the total differential values and divide them by
        all_diffs = []
        for c in avg_hist_diff_fighters.loc[fighter_index].iteritems():
            stat = c[1].values[0]
            col = c[0]
            if "total_" in col:
                if fighter_counted_fights == 0:
                    all_diffs.append(stat)
                else:
                    all_diffs.append(stat / fighter_counted_fights)

        # Get the historical fighter diff stats column names
        cols = []
        for x in avg_hist_diff_fights:
            if "historical" in x:
                cols.append(x)

        avg_hist_diff_fights.loc[row.Index, cols] = all_diffs

        # Add to the counted fights
        avg_hist_diff_fighters.loc[fighter_index, "counted_fights"] += 1

        # Add to fighter stats (row values past enum index of 34 are differential stats)
        i = 16  # fighter diff stat index starts 16, 36 is end
        for idx, s in enumerate(row):
            if idx > 34:  # fight differential stats index start, ends at 55
                avg_hist_diff_fighters.iloc[fighter_index, i] += s
                i += 1

        print(row.Index)

    return avg_hist_diff_fighters

def main():

    # Create Elo scorer
    elo_scorer = Elo(k=46)  # k = uncertainty, in chess 10 is used at high level, 40 at teen level

    # Update data
    fighter_df, hist_df = update_data()

    # Invert the data so older fights are first
    hist_df = hist_df.iloc[::-1]

    # Clean the fight history of draws/NCs
    no_draw_hist_df = remove_draws(hist_df)

    # Add second tier variables
    # accuracy?
    # Get a dictionary of row name : index number so we can call the index number in a for loop
    new_vars_hist_df = no_draw_hist_df.copy()

    # Calculate differential stats
    differential_stats = get_differential_stats(no_draw_hist_df)

    # Update the differntial stats to the new dataframe
    new_vars_hist_df = update_diff_stats(new_vars_hist_df, no_draw_hist_df, differential_stats)

    # Remove loser lines (loser lines are just mirror data of winner lines)
    unmirrored_hist_df = no_draw_hist_df[no_draw_hist_df.result == 1]

    # Add fighters to ELO players
    elo_scorer = add_fighters(elo_scorer, unmirrored_hist_df)

    # Create fighter df with diff avg column and counted fights so we know how much to divide the diff avg from
    avg_hist_diff_fighters = fighter_df.copy()
    avg_hist_diff_fighters.loc[:, "counted_fights"] = 0

    # Create fight df with fight stats, diff stats, and historical avg diff up to that fight
    avg_hist_diff_fights, avg_hist_diff_fighters = create_diff_dfs(new_vars_hist_df, avg_hist_diff_fighters)

    # Add differential stats for each fight
    avg_hist_diff_fighters = add_hist_diff_stats(new_vars_hist_df, avg_hist_diff_fights, avg_hist_diff_fighters)

    # Prediction value for each differential stat at time of fight
    b = hist_diff_brier_score(avg_hist_diff_fights)

    # Calculate brier score for each variable
    #brier_score = calc_brier_score(new_vars_hist_df)

    # Run Elo calc
    elo_scorer, historical_elo, monthly_top25, top25 = run_elo_calc(unmirrored_hist_df, elo_scorer)

    # > monthly_top25["6/2010"]
    # > top25
    # > elo_scorer["Anderson Silva"]
    # > peak_elo(historical_elo, "Anderson Silva")
    # Prediction value for each differential stat at time of fight
    # > b

    embed()

if __name__ == "__main__":
    main()