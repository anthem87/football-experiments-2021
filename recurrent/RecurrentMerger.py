import pandas as pd


class RecurrentMerger:

    def __init__(self):
        self.games = pd.DataFrame()
        self.all_games = []
        print('Merging...')

    def load_excels(self, paths):
        counter = 0
        for path in paths:
            self.all_games.append(pd.read_excel(path))
            print('Loaded file number ' + str(counter))
            counter += 1
        print('Merging all files...')
        self.merge_all()
        print('Converting team to categorical...')
        self.team_to_categorical()
        print('Converting scheme to categorical...')
        self.scheme_to_categorical()
        print('Calculating score differential...')
        self.calculate_score_differential()
        print('Calculating classes...')
        self.calculate_class()
        print('Dropping goals...')
        self.drop_goals()
        print('Creating new features...')
        self.create_new_features()
        print('Saving file...')
        self.save_to_excel(self.games)

    def merge_all(self):
        self.games = pd.concat(self.all_games)

    def drop_goals(self):
        self.games.drop('Goals_x', axis=1, inplace=True)
        self.games.drop('Goals_y', axis=1, inplace=True)
        self.games.drop('Conceded goals_x', axis=1, inplace=True)
        self.games.drop('Conceded goals_y', axis=1, inplace=True)

    def team_to_categorical(self):
        self.games['Team_x'] = pd.Categorical(self.games['Team_x'].astype('category')).codes
        self.games['Team_y'] = pd.Categorical(self.games['Team_y'].astype('category')).codes

    def scheme_to_categorical(self):
        self.games['Scheme_x'] = pd.Categorical(self.games['Scheme_x'].astype('category')).codes
        self.games['Scheme_y'] = pd.Categorical(self.games['Scheme_y'].astype('category')).codes

    def create_new_features(self):
        self.games.insert(5, 'Defensive duels lost_x',
                          self.games['Defensive duels_x'] - self.games['Defensive duels won_x'], True)
        self.games.insert(6, 'Shots conceded per duels lost_x',
                          self.games['Shots_y'] / self.games['Defensive duels lost_x'])

        self.games.insert(7, 'Defensive duels lost_y',
                          self.games['Defensive duels_y'] - self.games['Defensive duels won_y'], True)
        self.games.insert(8, 'Shots conceded per duels lost_y',
                          self.games['Shots_x'] / self.games['Defensive duels lost_y'])

    def save_to_excel(self, dataframe):
        path = './recurrent-out/recurrent-out.xlsx'
        writer_orig = pd.ExcelWriter(path, engine='xlsxwriter')
        dataframe.to_excel(writer_orig, index=False, sheet_name='result')
        writer_orig.save()

    def calculate_score_differential(self):
        self.games['Score-Differential'] = self.games['Goals_x'] - self.games['Goals_y']

    def calculate_class(self):
        self.games.loc[self.games['Score-Differential'] > 0, 'Final-Result'] = 'HOME'
        self.games.loc[self.games['Score-Differential'] == 0, 'Final-Result'] = 'DRAW'
        self.games.loc[self.games['Score-Differential'] < 0, 'Final-Result'] = 'AWAY'
