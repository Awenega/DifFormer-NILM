from collections import defaultdict
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from Utils.timefeatures_utils import time_features
from sklearn.preprocessing import StandardScaler
import joblib
import os


class UK_DALE_Dataset(Dataset):
    def __init__(self, period=None):
        super(UK_DALE_Dataset, self).__init__()

        cutoff = {
            'aggregate': 6000,
            'kettle': 3100,
            'fridge': 300,
            'washing_machine': 2500,
            'microwave': 3000
        }
        threshold = {
            'kettle': 2000,
            'fridge': 50,
            'washing_machine': 20,
            'microwave': 200
        }
        min_on = {
            'kettle': 2,
            'fridge': 10,
            'washing_machine': 300,
            'microwave': 2
        }
        min_off = {
            'kettle': 0,
            'fridge': 2,
            'washing_machine': 26,
            'microwave': 5
        }
        self.house_indicies = [1, 3, 4, 5] if period == 'train' else [2]
        self.appliance_names = ['washing_machine']
        self.sampling = '6s'
        self.cutoff = [cutoff[i] for i in ['aggregate'] + self.appliance_names]
        self.threshold = [threshold[i] for i in self.appliance_names]
        self.min_on = [min_on[i] for i in self.appliance_names]
        self.min_off = [min_off[i] for i in self.appliance_names]
        self.agg, self.app, timestamp = self.load_data()
        self.time_features = time_features(timestamp, freq='6S')[:, [2,3,4]]
        self.dataset = np.concatenate((self.app, self.time_features, self.agg), axis=-1)
        if period == 'train':
            self.scaler_agg = StandardScaler()
            self.scaler_app = StandardScaler()
            self.scaler_agg.fit(self.agg)
            self.scaler_app.fit(self.app)
            if not os.path.exists('scaler'):
                os.makedirs('scaler')
            joblib.dump(self.scaler_agg, 'scaler/scaler_agg.pkl')
            joblib.dump(self.scaler_app, 'scaler/scaler_app.pkl')
        else:
            self.scaler_agg = joblib.load('scaler/scaler_agg.pkl')
            self.scaler_app = joblib.load('scaler/scaler_app.pkl')
        
    def __len__(self):
        return len(self.dataset)
    
    def load_data(self):
        directory = Path('Data').joinpath('uk_dale')

        entire_data = None
        for house_id in self.house_indicies:
            house_folder = directory.joinpath('house_' + str(house_id))
            house_label = pd.read_csv(house_folder.joinpath('labels.dat'), sep=' ', header=None)

            house_data = pd.read_csv(house_folder.joinpath('channel_1.dat'), sep=' ', header=None)
            house_data.iloc[:, 0] = pd.to_datetime(house_data.iloc[:, 0], unit='s')
            house_data.columns = ['time', 'aggregate']
            house_data = house_data.set_index('time')
            house_data = house_data.resample(self.sampling).mean().fillna(method='ffill', limit=30)

            appliance_list = house_label.iloc[:, 1].values
            app_index_dict = defaultdict(list)

            for appliance in self.appliance_names:
                data_found = False
                for i in range(len(appliance_list)):
                    if appliance_list[i] == appliance:
                        app_index_dict[appliance].append(i + 1)
                        data_found = True

                if not data_found:
                    app_index_dict[appliance].append(-1)

            if np.sum(list(app_index_dict.values())) == -len(self.appliance_names):
                self.house_indicies.remove(house_id)
                continue

            for appliance in self.appliance_names:
                if app_index_dict[appliance][0] == -1:
                    house_data.insert(len(house_data.columns), appliance, np.zeros(len(house_data)))
                else:
                    temp_data = pd.read_csv(house_folder.joinpath('channel_' + str(app_index_dict[appliance][0]) + '.dat'), sep=' ', header=None)
                    temp_data.iloc[:, 0] = pd.to_datetime(temp_data.iloc[:, 0], unit='s')
                    temp_data.columns = ['time', appliance]
                    temp_data = temp_data.set_index('time')
                    temp_data = temp_data.resample(self.sampling).mean().fillna(method='ffill', limit=30)
                    house_data = pd.merge(house_data, temp_data, how='inner', on='time')

            if house_id == self.house_indicies[0]:
                entire_data = house_data
                if len(self.house_indicies) == 1:
                    entire_data = entire_data.reset_index(drop=True)
            else:
                entire_data = pd.concat([entire_data, house_data], ignore_index=True)
        
        entire_data = entire_data.dropna().copy()
        
        entire_data = entire_data[entire_data['aggregate'] > 0]
        entire_data[entire_data < 5] = 0
        entire_data = entire_data.clip([0] * len(entire_data.columns), self.cutoff, axis=1)
        timestamp = (entire_data.index.astype('int64') // 10**9)            
        return entire_data.values[:, 0].reshape(-1, 1), entire_data.values[:, 1:], timestamp.to_numpy()

def check_metrics():
    ground_truth = np.load("OUTPUT/ukdale/samples/ukdale_ground_truth_256_test.npy")[:, :, :1]
    fake = np.load("OUTPUT/ukdale/ddpm_fake_ukdale_unnormalized.npy")
    print(ground_truth.shape, fake.shape)

    mae = np.mean(np.abs(fake - ground_truth))
    print(f"MAE: {mae}")