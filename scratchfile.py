class FightData():
    def __init__(self):
        self.fight_hist = "C:\\Users\\dan\\PycharmProjects\\mma\\data\\fight_hist.csv"
        self.fight_stats = "C:\\Users\\dan\\PycharmProjects\\mma\\data\\fighter_stats.csv"

    def update(self):
        """
        Returns the updated fight history and fighter stats dataframes
        """
        # get old data
        fight_hist_old = pd.read_csv(self.fight_hist)
        fighter_stats_old = pd.read_csv(self.fight_stats)

        # update fight history
        fight_hist_updated = update_fight_stats(fight_hist_old)

        # update fighter stats
        fighter_stats_updated = update_fighter_details(fight_hist_updated.fighter_url.unique(), fighter_stats_old)

        return fight_hist_updated, fighter_stats_updated

    def write(self, fight_hist_updated, fighter_stats_updated):
        """
        Write data to csv
        """
        # save updated dataframes to file
        fight_hist_updated.to_csv(self.fight_hist, index = False)
        fighter_stats_updated.to_csv(self.fight_stats, index = False)


# Links
https://medium.com/geekculture/ranking-mma-fighters-using-the-elo-rating-system-2704adbf0c94
https://www.kaggle.com/calmdownkarm/ufc-predictor-and-notes

# [2:] prevents win corr with itself and win corr with round
c = no_draw_hist_df.corr()["result"][2:].round(5)
# heatmap it, the double bracket reshapes it into the right size
c = no_draw_hist_df.corr()[["result"]]
b = sns.heatmap(c, annot=True, fmt="g", cmap='viridis')
plt.show()

#Sort correlation
c = new_vars_hist_df.corr()[["result"]].sort_values("result", ascending=False).round(8)

brier_score_eq = {k: v for k, v in sorted(brier_score.items(), key=lambda item: item[1])}
brier_score_output = \
{'head_strikes_landed_differential': 0.1535420666907371,
 'sig_strikes_landed_differential': 0.15440052505608137,
 'total_strikes_landed_differential': 0.16684946969506367,
 'ground_strikes_landed_differential': 0.18356124206225843,
 'total_strikes_attempts_differential': 0.1859257823948968,
 'ground_strikes_attempts_differential': 0.18901361186811344,
 'sig_strikes_attempts_differential': 0.1931577073516428,
 'head_strikes_attempts_differential': 0.1970488120785045,
 'ground_strikes_landed': 0.20879437316539054,
 'ground_strikes_attempts': 0.21336574389522678,
 'pass_differential': 0.21715630918323986,
 'knockdowns': 0.2206966340314761,
 'distance_strikes_landed_differential': 0.22201280133890522,
 'total_strikes_landed': 0.22525458006270296,
 'pass': 0.2293044805515032,
 'takedowns_landed_differential': 0.23098213182683555,
 'head_strikes_landed': 0.2315946918306528,
 'sig_strikes_landed': 0.23274713186535176,
 'takedowns_landed': 0.235624273098802,
 'clinch_strikes_landed_differential': 0.2375601558704735,
 'body_strikes_landed_differential': 0.2376657040052689,
 'clinch_strikes_attempts_differential': 0.24009270848302297,
 'sub_attempts': 0.2404508501707163,
 'total_strikes_attempts': 0.24047504495809233,
 'knockdowns_differential': 0.24387098915478775,
 'body_strikes_attempts_differential': 0.24389680178946874,
 'sub_attempts_differential': 0.2440893869894121,
 'head_strikes_attempts': 0.2450754112106454,
 'clinch_strikes_landed': 0.24580552903517047,
 'distance_strikes_landed': 0.24590036977166851,
 'body_strikes_landed': 0.24638102366003264,
 'clinch_strikes_attempts': 0.2466824075220038,
 'distance_strikes_attempts_differential': 0.2467464035968645,
 'sig_strikes_attempts': 0.24714964423567115,
 'takedowns_attempts_differential': 0.24800733308835576,
 'body_strikes_attempts': 0.24809686558682328,
 'leg_strikes_landed': 0.2484269483369769,
 'leg_strikes_landed_differential': 0.24845707173917594,
 'leg_strikes_attempts_differential': 0.2488781210130024,
 'takedowns_attempts': 0.24951650744446785,
 'distance_strikes_attempts': 0.24965223054889166,
 'leg_strikes_attempts': 0.25020488798707824}

correlation
ground_strikes_landed_differential      0.366250
head_strikes_landed_differential        0.350861
ground_strikes_attempts_differential    0.350042
ground_strikes_landed                   0.349596
ground_strikes_attempts                 0.339408
sig_strikes_landed_differential         0.320490
knockdowns                              0.314140
pass_differential                       0.288768
total_strikes_landed                    0.280435
total_strikes_landed_differential       0.273722
pass                                    0.268515
head_strikes_landed                     0.267821
takedowns_landed_differential           0.253893
head_strikes_attempts_differential      0.241788
sig_strikes_attempts_differential       0.241037
takedowns_landed                        0.240375
distance_strikes_landed_differential    0.240207
sig_strikes_landed                      0.240010
total_strikes_attempts_differential     0.233791
total_strikes_attempts                  0.187586
clinch_strikes_landed_differential      0.187260
body_strikes_landed_differential        0.182374
sub_attempts                            0.170395
knockdowns_differential                 0.165951
clinch_strikes_attempts_differential    0.163967
body_strikes_attempts_differential      0.142654
clinch_strikes_landed                   0.133630
head_strikes_attempts                   0.129801
sig_strikes_attempts                    0.129710
body_strikes_landed                     0.129136
sub_attempts_differential               0.128094
distance_strikes_landed                 0.118644
clinch_strikes_attempts                 0.114591
body_strikes_attempts                   0.093274
distance_strikes_attempts_differential  0.092817
leg_strikes_landed_differential         0.091076
takedowns_attempts_differential         0.088474
takedowns_attempts                      0.086023
leg_strikes_landed                      0.075877
leg_strikes_attempts_differential       0.066589
leg_strikes_attempts                    0.059770
distance_strikes_attempts               0.050142
round                                   0.000000


