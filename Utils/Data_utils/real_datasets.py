import os
import torch
import numpy as np

from torch.utils.data import Dataset
from Utils.Data_utils.uk_dale_utils import UK_DALE_Dataset


class CustomDataset(Dataset):
    def __init__(
        self, 
        name, 
        window=64, 
        proportion=0.8, 
        save2npy=True, 
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        strides=0,
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'

        self.name, self.window, self.stride,  = name, window, strides
        self.period, self.save2npy, self.dir = period, save2npy, os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        ds = UK_DALE_Dataset(period=period)
        self.rawdata, self.scaler_agg, self.scaler_app = ds.dataset, ds.scaler_agg, ds.scaler_app 
        self.var_num = self.rawdata.shape[-1]

        self.data = self.__normalize(self.rawdata)
        train, inference = self.__getsamples(self.data, proportion, seed)
        self.samples = train if period == 'train' else inference
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed):
        num_windows = (len(data) - self.window) // self.stride + 1
        x = np.zeros((num_windows, self.window, self.var_num))
        for i in range(num_windows):
            start = i * self.stride
            end = start + self.window

            x[i, :, :] = data[start:end, :]

        train_data, test_data = self.divide(x, proportion, seed)
        if self.save2npy and self.period == 'train':
            np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.__unnormalize(train_data))
        elif self.save2npy and self.period == 'test':
            np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.__unnormalize(test_data))
        return train_data, test_data
    
    def __normalize(self, rawdata):
        app_normalized = self.scaler_app.transform(rawdata[:,0].reshape(-1, 1))
        agg_normalized = self.scaler_agg.transform(rawdata[:,4].reshape(-1, 1))
        data = np.concatenate((app_normalized, self.rawdata[:, 1:4], agg_normalized), axis = -1)
        return data

    def __unnormalize(self, data):
        data = data.reshape(-1, self.var_num)
        app_not_normalized = self.scaler_app.inverse_transform(data[:, 0].reshape(-1,1))
        agg_not_normalized = self.scaler_agg.inverse_transform(data[:, 4].reshape(-1,1))
        data = np.concatenate((app_not_normalized, data[:, 1:4], agg_not_normalized), axis = -1)
        return data.reshape(-1, self.window, self.var_num)
      
    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.random.permutation(size)

        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]
        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    def __getitem__(self, ind):
        samples = self.samples[ind, :, :]
        return torch.from_numpy(samples).float()
                
    def __len__(self):
        return self.sample_num
    