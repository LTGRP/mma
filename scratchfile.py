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
b = sns.heatmap(c, annot=True, fmt="g", cmap='viridis')mp.show()

#Sort correlation
c = new_vars_hist_df.corr()[["result"]].sort_values("result", ascending=False).round(8)

# See historically largest elo
h = []
for x in historical_elo:
    for y in historical_elo[x]:
        elo = y[1]
        name = x
        h.append((name, elo))
h.sort(key=lambda z: z[1])


brier_score_eq = {k: v for k, v in sorted(brier_score.items(), key=lambda item: item[1])}
brier_score_output = \
{'head_strikes_landed_differential': 0.14943366168949745,
 'sig_strikes_landed_differential': 0.15136372295375505,
 'total_strikes_landed_differential': 0.1559420335007692,
 'total_strikes_attempts_differential': 0.18135058814178381,
 'ground_strikes_landed_differential': 0.18304686145901417,
 'ground_strikes_attempts_differential': 0.18669310881417725,
 'sig_strikes_attempts_differential': 0.1904116230302404,
 'head_strikes_attempts_differential': 0.20137211969018237,
 'ground_strikes_landed': 0.20597877034310552,
 'ground_strikes_attempts': 0.21154433723263055,
 'distance_strikes_landed_differential': 0.2211012578434788,
 'knockdowns': 0.22169781365368596,
 'head_strikes_landed': 0.22864909205334744,
 'total_strikes_landed': 0.2296788051905403,
 'takedowns_landed_differential': 0.23164476413526475,
 'takedowns_landed': 0.23325326533002208,
 'sig_strikes_landed': 0.23598530062579554,
 'total_strikes_attempts': 0.23892418768345777,
 'sub_attempts': 0.2398577861044575,
 'body_strikes_landed_differential': 0.240069966888866,
 'clinch_strikes_attempts_differential': 0.24024716343467087,
 'knockdowns_differential': 0.24081013699655604,
 'clinch_strikes_landed_differential': 0.2415314023168161,
 'clinch_strikes_landed': 0.24488275055601144,
 'body_strikes_attempts_differential': 0.24489809918989408,
 'sig_strikes_attempts': 0.24510194840874505,
 'sub_attempts_differential': 0.24541013253356064,
 'clinch_strikes_attempts': 0.2460430148878946,
 'body_strikes_landed': 0.246219692531269,
 'leg_strikes_landed_differential': 0.24753219896157821,
 'distance_strikes_attempts_differential': 0.2478353701709483,
 'leg_strikes_landed': 0.24793025740347663,
 'distance_strikes_landed': 0.24798888646464132,
 'takedowns_attempts': 0.2481873443543807,
 'body_strikes_attempts': 0.24820899591692222,
 'takedowns_attempts_differential': 0.24883153509793335,
 'head_strikes_attempts': 0.2489605633173813,
 'distance_strikes_attempts': 0.24908720711416965,
 'leg_strikes_attempts_differential': 0.24926103976519937,
 'leg_strikes_attempts': 0.24937371649779105,
 'pass_differential': 0.2501389635919092,
 'pass': 0.2502037492343585}

.corr() correlation
ground_strikes_landed_differential      0.366390
head_strikes_landed_differential        0.351367
ground_strikes_attempts_differential    0.350278
ground_strikes_landed                   0.349744
ground_strikes_attempts                 0.339564
sig_strikes_landed_differential         0.320761
knockdowns                              0.314320
total_strikes_landed                    0.280910
total_strikes_landed_differential       0.273236
head_strikes_landed                     0.268093
takedowns_landed_differential           0.254087
head_strikes_attempts_differential      0.242299
sig_strikes_attempts_differential       0.240814
sig_strikes_landed                      0.240410
takedowns_landed                        0.240389
distance_strikes_landed_differential    0.239981
total_strikes_attempts_differential     0.233659
clinch_strikes_landed_differential      0.188320
total_strikes_attempts                  0.187949
body_strikes_landed_differential        0.182988
sub_attempts                            0.169857
knockdowns_differential                 0.165670
clinch_strikes_attempts_differential    0.164458
body_strikes_attempts_differential      0.143152
clinch_strikes_landed                   0.134039
head_strikes_attempts                   0.129995
sig_strikes_attempts                    0.129970
body_strikes_landed                     0.129682
sub_attempts_differential               0.127669
distance_strikes_landed                 0.118837
clinch_strikes_attempts                 0.114919
body_strikes_attempts                   0.093753
distance_strikes_attempts_differential  0.092653
leg_strikes_landed_differential         0.091145
takedowns_attempts_differential         0.088399
takedowns_attempts                      0.085927
leg_strikes_landed                      0.076051
leg_strikes_attempts_differential       0.066489
leg_strikes_attempts                    0.059856
distance_strikes_attempts               0.050295
pass                                    0.026410
pass_differential                       0.019454
round                                  -0.000000

Strikes landed per min / strikes absorbed per min
sorted([(x,3),(y,2)], key=lambda x: x[1])

[('Tom Aspinall', 3.0337552742616034),
 ('Uros Medic', 3.0789473684210527),
 ('Gegard Mousasi', 3.0991735537190084),
 ('Dave Galera', 3.1509433962264146),
 ('Maiquel Falcao', 3.1509433962264146),
 ('Mike Jackson', 3.195488721804511),
 ('Cristiane Justino', 3.2355555555555555),
 ('Tatiana Suarez', 3.324137931034483),
 ('Mike King', 3.4418604651162794),
 ('Mike Guymon', 3.481012658227848),
 ('Ronnie Lawrence', 3.4862385321100913),
 ('Jesse Taylor', 3.527272727272727),
 ('Umar Nurmagomedov', 3.567901234567901),
 ('Alex Pereira', 3.649122807017544),
 ('Johnny Rees', 3.916666666666667),
 ('Mike Nickels', 4.0),
 ('Adlan Amagov', 4.01219512195122),
 ('Carlston Harris', 4.149606299212598),
 ('Ottman Azaitar', 4.7897727272727275),
 ('Keith Rockel', 5.3809523809523805),
 ('Ebenezer Fontes Braga', 5.761904761904762),
 ('Khalid Murtazaliev', 11.999999999999998),
 ('Tom Murphy', 14.705882352941176),
 ('Abdul-Kerim Edilov', 78.61538461538461),
 ('Khamzat Chimaev', 108.5)]

)
