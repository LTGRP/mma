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