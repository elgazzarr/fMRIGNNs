from torch_geometric.data import Data
import pandas as pd
import numpy as np
from torchvision.transforms import Compose
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import torch
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Pool
import torch_geometric.transforms as T

DATASETS = [
    'Mddrest',
    'Abide', 'Ukbb']

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def prop_thresholding(x, th):
    size = x.shape[0]**2
    x_flattend = x.flatten()
    topk = int(th * size)
    x_sorted = x_flattend[np.argsort(x_flattend)[::-1]]
    th_value = x_sorted[topk]
    rows, columns = np.where(x > th_value)
    return rows, columns


class Mddrest(Dataset):
    LABEL = 'Diagnosis'

    def __init__(self,
                 df_dir,th, train, use_gdc=True):
        super(Mddrest).__init__()


        # Read the parent CSV file
        data_info = df_dir
        data_info = data_info.sample(frac=1).reset_index(drop=True)
        self.num_classes = 2

        #Determine the n-channels (=nrois) from the data by using the first sample
        sample_file = data_info['tc_atlas'].iloc[0].replace('ATLAS', 'HO_112')
        ntime, nrois = np.load(sample_file).shape
        self.nrois = nrois
        self.ntime = 180
        self.total_subjects = len(data_info)
        self.labels = []
        self.graphs = []
        self.cc = []
        self.train = train
        self.augment = False
        self.crop_len = 60
        self.crop = False
        self.augmenter = (TimeWarp() @ 0.5
         + Crop(size=self.crop_len)
         + Drift(max_drift=(0.1, 0.5)) @ 0.2
         + Reverse() @ 0.2)

        for i, sub_i in enumerate(data_info.index):

                tc_file = data_info['tc_atlas'].iloc[i].replace('ATLAS', 'HO_112')
                tc_vals = np.load(tc_file).transpose()

                if tc_vals.shape[1] < self.ntime:
                    tc_vals = np.array([np.pad(tc_vals[i],(0,self.ntime-tc_vals.shape[1]),'reflect')
                                    for i in range(tc_vals.shape[0])])
                else:
                    tc_vals = tc_vals[:,:self.ntime]

                tc_vals = np.expand_dims(np.expand_dims(tc_vals,0),0)
                cc_file = data_info['cc_file'].iloc[i].replace('ATLAS', 'HO_112')
                cc_vals = np.load(cc_file)
                cc_triu_ids = np.triu_indices(nrois)
                cc_vector = np.expand_dims(cc_vals[cc_triu_ids],0)

                x = torch.tensor(cc_vals, dtype=torch.float)
                t = torch.tensor(tc_vals, dtype=torch.float)
                x_flattend = torch.tensor(cc_vector, dtype=torch.float)

                rows, columns = prop_thresholding(np.abs(cc_vals), th) #np.where(np.abs(cc_vals)>th)
                edge_inds = torch.tensor(np.column_stack((rows,columns)).transpose(),dtype=torch.long)

                graph = Data(x=x, x_flat=x_flattend, t=t, edge_index=edge_inds, y=data_info[self.LABEL].iloc[i])


                if use_gdc:
                    transform = T.GDC(
                        self_loop_weight=0,
                        normalization_in='sym',
                        normalization_out='col',
                        diffusion_kwargs=dict(method='heat', t=10),
                        sparsification_kwargs=dict(method='threshold', eps=0.9),
                        exact=True,
                    )
                    graph = transform(graph)


                self.labels.append(graph.y)
                self.graphs.append(graph)
                self.cc.append(cc_vals)

    def __len__(self):
        return self.total_subjects

    def __getitem__(self, index):
        data = self.graphs[index]



        return self.graphs[index]

    def __getallitems__(self):
        return self.graphs

class Abide(Dataset):
    LABEL = 'Diagnosis'

    def __init__(self,
                 df_dir,th, train, use_gdc=False):
        super(Abide).__init__()


        print('Reading csv file...')
        # Read the parent CSV file
        data_info = df_dir
        data_info = data_info.sample(frac=1).reset_index(drop=True)
        self.num_classes = 2

        #Determine the n-channels (=nrois) from the data by using the first sample
        sample_file = data_info['tc_file'].iloc[0].replace('ATLAS', 'craddock_200')
        ntime, nrois = pd.read_csv(sample_file).values.shape
        self.nrois = nrois
        self.ntime = 150
        self.total_subjects = len(data_info)
        self.labels = []
        self.graphs = []
        self.cc = []

        for i, sub_i in enumerate(data_info.index):

                tc_file = data_info['tc_file'].iloc[i].replace('ATLAS', 'craddock_200')
                tc_vals = pd.read_csv(tc_file).values.transpose()
                if tc_vals.shape[1] < self.ntime:
                    tc_vals = np.array([np.pad(tc_vals[i],(0,self.ntime-tc_vals.shape[1]),'reflect')
                                    for i in range(tc_vals.shape[0])])
                else:
                    tc_vals = tc_vals[:,:self.ntime]

                tc_vals = np.expand_dims(np.expand_dims(tc_vals,0),0)

                cc_file = data_info['cc_file'].iloc[i].replace('ATLAS', 'craddock_200')
                cc_vals = np.load(cc_file)
                cc_triu_ids = np.triu_indices(nrois)
                cc_vector = np.expand_dims(cc_vals[cc_triu_ids],0)

                x = torch.tensor(cc_vals, dtype=torch.float)
                t = torch.tensor(tc_vals, dtype=torch.float)
                x_flattend = torch.tensor(cc_vector, dtype=torch.float)

                rows, columns = prop_thresholding(cc_vals, th) #np.where(np.abs(cc_vals)>th)
                edge_inds = torch.tensor(np.column_stack((rows,columns)).transpose(),dtype=torch.long)

                graph = Data(x=x, x_flat=x_flattend, t=t, edge_index=edge_inds, y=data_info[self.LABEL].iloc[i])

                if use_gdc:
                    transform = T.GDC(
                        self_loop_weight=0,
                        normalization_in='sym',
                        normalization_out='col',
                        diffusion_kwargs=dict(method='heat', t=10),
                        sparsification_kwargs=dict(method='threshold', eps=0.9),
                        exact=True,
                    )
                    graph = transform(graph)

                self.labels.append(graph.y)
                self.graphs.append(graph)
                self.cc.append(cc_vals)

    def __len__(self):
        return self.total_subjects

    def __getitem__(self, index):
        return self.graphs[index]

    def __getallitems__(self):
        return self.graphs


class Ukbb(Dataset):
    LABEL = 'Sex'

    def __init__(self,
                 df_dir,th, train, use_gdc=True):
        super(Ukbb).__init__()

        print('Reading csv file...')
        # Read the parent CSV file
        data_info = df_dir
        data_info = data_info.sample(frac=1).reset_index(drop=True)
        self.num_classes = 2

        #Determine the n-channels (=nrois) from the data by using the first sample
        sample_file = data_info['tc_file'].iloc[0].replace('ATLAS', 'AAL')
        nrois = pd.read_csv(sample_file).values.shape[1] - 1
        self.nrois = nrois
        self.ntime = 490
        self.total_subjects = len(data_info)
        self.labels = []
        self.graphs = []
        self.cc = []

        for i, sub_i in enumerate(data_info.index):

                tc_file = data_info['tc_file'].iloc[i].replace('ATLAS', 'AAL')
                tc_vals = pd.read_csv(tc_file).values.transpose()[1:, :self.ntime]

                corr_file = data_info['corrmat_file'].iloc[i].replace('ATLAS', 'AAL')
                cc_vals = np.load(corr_file)
                cc_triu_ids = np.triu_indices(nrois)
                cc_vector = np.expand_dims(cc_vals[cc_triu_ids],0)
                tc_vals = np.expand_dims(np.expand_dims(tc_vals,0),0)


                x = torch.tensor(cc_vals, dtype=torch.float)
                x_flattend = torch.tensor(cc_vector, dtype=torch.float)
                t = torch.tensor(tc_vals, dtype=torch.float)

                rows, columns = np.where(cc_vals>th)
                edge_inds = torch.tensor(np.column_stack((rows,columns)).transpose(),dtype=torch.long)

                graph = Data(x=x, x_flat=x_flattend, t=t, edge_index=edge_inds, y = 0 if data_info[self.LABEL].iloc[i] =='Male' else 1)
                if use_gdc:
                    transform = T.GDC(
                        self_loop_weight=0,
                        normalization_in='sym',
                        normalization_out='col',
                        diffusion_kwargs=dict(method='heat', t=5),
                        sparsification_kwargs=dict(method='threshold', eps=0.7),
                        exact=True,
                    )
                    graph = transform(graph)
                self.labels.append(graph.y)
                self.graphs.append(graph)
                self.cc.append(cc_vals)

    def __len__(self):
        return self.total_subjects

    def __getitem__(self, index):
        return self.graphs[index]

    def __getallitems__(self):
        return self.graphs
