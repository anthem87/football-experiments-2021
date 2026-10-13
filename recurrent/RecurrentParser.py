import pandas as pd
import numpy as np
import os
import random


class RecurrentParser:

    def __init__(self):
        self.games = pd.DataFrame()
        self.home_team = ''
        self.path = ''

    def parse_excel(self, path):
        self.path = path
        self.games = pd.read_excel(path)
        self.drop_first_data()
        self.date_to_float()
        self.parse_scheme()
        self.save_home_team()
        self.rename_columns()
        merged = self.merge_home_away()
        merged = self.adjust_rows(merged)
        self.drop_duration_away(merged)
        self.save(merged)

    def adjust_rows(self, dataframe):
        number_of_rows = len(dataframe.index) - 1
        if number_of_rows < 4:
            print('Found dataframe count less than 4, removing file...')
            os.remove(path=self.path)
        else:
            while (len(dataframe.index - 1) % 4) != 0:
                print('rimuovo il resto di: ' + str(len(dataframe.index - 1) % 4))
                dataframe.drop(dataframe.tail(1).index, inplace=True)
        # print((len(dataframe.index) - 1) % 4)
        return dataframe

    def is_multiple(self, x, y):
        return x != 0 and y % x == 0

    def drop_first_data(self):
        self.games.drop('Match', axis=1, inplace=True)
        self.games.drop('Competition', axis=1, inplace=True)
        self.games.drop([0, 1], inplace=True)
        self.games.reset_index(drop=True, inplace=True)

    def date_to_float(self):
        # transform date to float
        self.games['Date'] = pd.to_datetime(self.games['Date']).values.astype(np.float32)

    def parse_scheme(self):
        self.games['Scheme'] = self.games['Scheme'].str.split(n=1, pat="(", expand=False).str[0]

    def save_home_team(self):
        self.home_team = self.games.at[0, 'Team']

    def merge_home_away(self):
        merged = pd.merge(self.games[self.games['Team'] == self.home_team],
                          self.games[self.games['Team'] != self.home_team], on='Date')
        return merged

    def drop_duration_away(self, merged_df):
        merged_df.drop('Duration_y', axis=1, inplace=True)

    def save(self, dataframe):
        path = './second-step-data-recurrent/' + self.home_team + '.xlsx'
        if os.path.isfile(path):
            number = random.randint(0, 22)
            path = path + str(number) + '.xlsx'
        writer_orig = pd.ExcelWriter(path, engine='xlsxwriter')
        dataframe.to_excel(writer_orig, index=False, sheet_name=self.home_team)
        writer_orig.save()

    def rename_columns(self):
        self.games.rename(columns={

            'Shots / on target': 'Shots',
            'Unnamed: 9': 'Shots on target',
            'Unnamed: 10': 'Shots on target Percent',
            'Passes / accurate': 'Passes',
            'Unnamed: 12': 'Accurate Passes',
            'Unnamed: 13': 'Acc. Passes Percent',
            'Possession, %': 'Possession Percent',
            'Long pass %': 'Long pass Percent',
            'Losses / Low / Medium / High': 'Losses',
            'Unnamed: 16': 'Losses Low',
            'Unnamed: 17': 'Losses Medium',
            'Unnamed: 18': 'Losses High',
            'Recoveries / Low / Medium / High': 'Recoveries',
            'Unnamed: 20': 'Recoveries Low',
            'Unnamed: 21': 'Recoveries Medium',
            'Unnamed: 22': 'Recoveries High',
            'Duels / won': 'Duels',
            'Unnamed: 24': 'Duels won',
            'Unnamed: 25': 'Duels won Percent',
            'Shots from outside penalty area / on target': 'Shots from outside penalty area',
            'Unnamed: 27': 'Outer shots on target',
            'Unnamed: 28': 'Outer shots on target Percent',
            'Positional attacks / with shots': 'Positional attacks',
            'Unnamed: 30': 'Positional attacks with shots',
            'Unnamed: 31': 'Positional attacks with shots Percent',
            'Counterattacks / with shots': 'Counterattacks',
            'Unnamed: 33': 'Counterattacks with shots',
            'Unnamed: 34': 'Counterattacks with shots Percent',
            'Set pieces / with shots': 'Set pieces',
            'Unnamed: 36': 'Set pieces with shots',
            'Unnamed: 37': 'Set pieces with shots Percent',
            'Corners / with shots': 'Corners',
            'Unnamed: 39': 'Corners with shots',
            'Unnamed: 40': 'Corners with shots Percent',
            'Free kick / with shots': 'Free kicks',
            'Unnamed: 42': 'Free kicks with shots',
            'Unnamed: 43': 'Free kicks with shots Percent',
            'Penalties / converted': 'Penalties',
            'Unnamed: 45': 'Penalties converted',
            'Unnamed: 46': 'Penalties converted Percent',
            'Crosses: / accurate': 'Crosses',
            'Unnamed: 48': 'Crosses accurate',
            'Unnamed: 49': 'Crosses accurate Percent',
            'Penalty area entries (runs / crosses)': 'Penalty area entries',
            'Unnamed: 53': 'Penalty area runs',
            'Unnamed: 54': 'Penalty area crosses',
            'Offensive duels / won': 'Offensive duels',
            'Unnamed: 57': 'Offensive duels won',
            'Unnamed: 58': 'Offensive duels won Percent',
            'Shots against / on target': 'Shots against',
            'Unnamed: 62': 'Shot against on target',
            'Unnamed: 63': 'Shot against on target Percent',
            'Defensive duels / won': 'Defensive duels',
            'Unnamed: 65': 'Defensive duels won',
            'Unnamed: 66': 'Defensive duels won Percent',
            'Aerial duels / won': 'Aerial duels',
            'Unnamed: 68': 'Aerial duels won',
            'Unnamed: 69': 'Aerial duels won Percent',
            'Sliding tackles / successful': 'Sliding tackles',
            'Unnamed: 71': 'Sliding tackles successful',
            'Unnamed: 72': 'Sliding tackles successful Percent',
            'Forward passes / accurate': 'Forward passes',
            'Unnamed: 79': 'Forward passes / accurate',
            'Unnamed: 80': 'Forward passes / accurate Percent',
            'Back passes / accurate': 'Back passes',
            'Unnamed: 82': 'Forward passes accurate',
            'Unnamed: 83': 'Forward passes accurate Percent',
            'Lateral passes / accurate': 'Lateral passes',
            'Unnamed: 85': 'Lateral passes accurate',
            'Unnamed: 86': 'Lateral passes accurate Percent',
            'Long passes / accurate': 'Long passes',
            'Unnamed: 88': 'Long passes accurate',
            'Unnamed: 89': 'Long passes accurate Percent',
            'Passes to final third / accurate': 'Passes to final third',
            'Unnamed: 91': 'Passes to final third accurate',
            'Unnamed: 92': 'Passes to final third accurate Percent',
            'Progressive passes / accurate': 'Progressive passes',
            'Unnamed: 94': 'Progressive passes accurate',
            'Unnamed: 95': 'Progressive passes accurate Percent',
            'Smart passes / accurate': 'Smart passes',
            'Unnamed: 97': 'Smart passes accurate',
            'Unnamed: 98': 'Smart passes accurate Percent',
            'Throw ins / accurate': 'Throw ins',
            'Unnamed: 100': 'Throw ins accurate',
            'Unnamed: 101': 'Throw ins accurate Percent'},
            inplace=True)

