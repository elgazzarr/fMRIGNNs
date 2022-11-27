from data import *
from train import train, test_model
from networks.baseline_models import *
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from main_dynamic import prepare_ukbb_loaders
import wandb
import warnings
import argparse
warnings.filterwarnings("ignore")



def run_1fold_ukbb(df_path, dataset,  model_name, config, device):
        model_hparams  = ["dropout", "dilation_exponential", "f1", "f2",  "k1",  "k2", 'readout']
        model_params =  {key: config[key] for key in model_hparams}

        dataset_name = dataset[0:4]
        N = dataset[5:]
        th = config['th']
        df = pd.read_csv(df_path)
        sample_dataset = get_dataset_class(dataset_name)(df[0:10], th, train = True)
        num_nodes = sample_dataset.nrois
        label = sample_dataset.LABEL

        mean_adj = None
        model = get_model_class(model_name)(nrois = num_nodes, **model_params)

        train_loader, val_loader, test_loader  = prepare_ukbb_loaders(dataset_name,N,config)
        trained_model = train(model, train_loader, val_loader, config, device)
        accs, senss, specs = test_model(model, test_loader, device)
        print(" Test Accuracy = {:.2f}% \n Test Sens = {:.2f}% \n Test Spec = {:.2f}%".format( accs*100 , senss *100, specs*100))

        wandb.log({'Test_acc':accs, 'Test_sens':senss, 'Test_spec': specs})

def run_kfold(df_path, dataset,  model_name, config,  device):

    th = config['th']
    df = pd.read_csv(df_path)
    skf = model_selection.StratifiedKFold(n_splits=5)
    skf.get_n_splits(df, df['stf'])

    accs = []
    senss = []
    specs = []
    k = 0

    model_hparams  = ["dropout", "dilation_exponential", "f1", "f2",  "k1",  "k2", 'readout']
    model_params =  {key: config[key] for key in model_hparams}

    for train_index, test_index in skf.split(df, df['stf']):

        k+=1

        train_val_df = df.iloc[train_index]
        train_df, val_df = train_test_split(train_val_df, test_size = 0.15, stratify = train_val_df['stf'], random_state=0)
        test_df = df.iloc[test_index]

        train_loader = DataLoader(get_dataset_class(dataset)(train_df, th, train = True), batch_size=config['batch_size'],   num_workers=5, pin_memory=False)
        val_loader = DataLoader(get_dataset_class(dataset)(val_df, th, train = False), batch_size=config['batch_size'],   num_workers=5, pin_memory=False)
        test_loader = DataLoader(get_dataset_class(dataset)(test_df, th, train = False), batch_size=config['batch_size'], num_workers=5, pin_memory=False)

        mean_adj = None
        num_nodes = 112 if dataset == 'Mddrest' else 195

        model = get_model_class(model_name)(nrois = num_nodes, **model_params)

        trained_model = train(model, train_loader, val_loader, config, device)
        acc, sens, spec = test_model(model, test_loader, device)

        accs.append(acc)
        senss.append(sens)
        specs.append(spec)
        print("for fold {}, Acc = {:.3f}, Sens = {:.3f}, Spec  = {:.3f}".format(k,acc,sens,spec))
        print('-'*30)

    accs = np.array(accs)
    acc_mean = np.round(np.mean(accs),3)
    acc_std = np.round(np.std(accs),3)
    senss = np.array(senss)
    sens_mean = np.round(np.mean(senss),3)
    sens_std = np.round(np.std(senss),3)
    specs = np.array(specs)
    spec_mean = np.round(np.mean(specs),3)
    spec_std = np.round(np.std(specs),3)

    print (f'{model_name} on {dataset} dataset 5-fold results:')
    print(" Test Accuracy: mean = {:.3f} % ,std = {:.3f}".format(acc_mean*100, acc_std*100))
    print(" Test Sens: mean = {:.3f} % ,std = {:.3f}".format(sens_mean*100, sens_std*100))
    print(" Test Spec: mean = {:.3f} % ,std = {:.3f}".format(spec_mean*100, spec_std*100))
    wandb.log({'Test_acc':acc_mean, 'Test_sens':sens_mean, 'Test_spec': spec_mean, 'std': [acc_std, sens_std, spec_std]})


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device',type=str,default='cuda:1',help='device')
    parser.add_argument('--dataset',type=str,default='Abide',help='Dataset name, valid options are ["Abide", "Mddrest", "UKBB_N"]')
    parser.add_argument('--model',type=str,default='MLP', help='Baseline model name. valid options are ["MLP", "CNN_1D"]')
    args = parser.parse_args()

    np.random.seed(0)
    device = torch.device(args.device)

    dataset = args.dataset
    model_name = args.model
    df_paths = {'Mddrest': '../csvfiles/mddrest.csv'  , 'Abide':'../csvfiles/abide.csv',  'Ukbb_500': '../csvfiles/ukbb_500.csv',
     'Ukbb_1000': '../csvfiles/ukbb_1000.csv', 'Ukbb_2000': '../csvfiles/ukbb_2000.csv', 'Ukbb_5000': '../csvfiles/ukbb_5000.csv' }
    df_path = df_paths[args.dataset]

    hparams_defaults = {"dropout":0.2, "dilation_exponential":2, "f1":1024, "f2":128, 'batch_size': 32,
     'lr': 1e-3, 'k1': 3,'k2': 3, 'ls': 0.1,'th':0.5, 'readout': 'mean'}


    config = hparams_defaults
    wandb.init(config = hparams_defaults, project= model_name+'-'+dataset)
    config = wandb.config



    print(f'Running a {model_name} on the {dataset} dataset.')

    print('*'*50)
    if dataset not in ['Abide', 'Mddrest']:
        run_1fold_ukbb(df_path, dataset,  model_name, config,  device)
    else:
        run_kfold(df_path, dataset,  model_name, config,  device)

    print('-'*30)
