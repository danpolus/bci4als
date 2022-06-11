import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
import os
import pickle
from tkinter import filedialog

from src.bci4als.ml_model import MLModel
from src.bci4als.experiments.offline import OfflineExperiment
from Session import SessionType

def offline_experiment(eeg, sessType: SessionType):

    num_trials = 60

    session_directory = "C:\My Files\Work\BGU\Datasets\drone BCI"
    if sessType == SessionType.OfflineTrainCspMI:
        in_fn = "\\train_data.mat"
    elif sessType == SessionType.OfflineTrainLdaMI:
        in_fn = "\\alpha_beta_aug15_augmented_source_data.mat"

    #######################################################

    if sessType == SessionType.OfflineExpMI:
        exp = OfflineExperiment(eeg=eeg, num_trials=num_trials, trial_length=eeg.epoch_len_sec, full_screen=True, audio=False)
        trials, labels = exp.run()
        train_data = {'trials':np.stack(trials), 'labels':labels}
        session_directory = exp.session_directory
        savemat(session_directory + "\\train_data.mat", train_data)
        model = train_model_save_source(trials, labels, eeg, session_directory)
        with open(session_directory+"\\model.pkl", 'wb') as file: #save model
            pickle.dump(model, file)

    elif sessType == SessionType.OfflineTrainCspMI or sessType == SessionType.OfflineTrainLdaMI:

        in_fn_list = []
        train_acc_list = []
        val_acc_list = []
        in_dir = filedialog.askdirectory(title='Select trials file', initialdir=session_directory)
        if os.path.exists(in_dir+in_fn):
            in_fn_list.append(in_dir+in_fn)
        else:
            in_dir_list = os.listdir(in_dir)
            for fp in in_dir_list:
                if os.path.isdir(in_dir+"\\"+fp) and os.path.exists(in_dir+"\\"+fp+in_fn):
                    in_fn_list.append(in_dir+"\\"+fp+in_fn)

        for fn in in_fn_list:
            session_directory = os.path.dirname(fn)

            if sessType == SessionType.OfflineTrainCspMI:
                cleanTrainData = load_augmented_data(fn)
                trials=cleanTrainData['trials']
                labels=cleanTrainData['labels']
                model = train_model_save_source(trials, labels, eeg, session_directory)

            else:
                model = pickle.load(open(session_directory+"\\model.pkl", 'rb')) #load model
                augSourceData = load_augmented_data(fn)
                source_data = np.stack([t.to_numpy().T for t in augSourceData['trials']])
                augmented_source_data = np.stack([t.to_numpy().T for t in augSourceData['augmented_trials']])
                model.class_train(source_data, augmented_source_data, augSourceData['labels'], augSourceData['augmented_labels'], eeg)
                # model.class_train(augmented_source_data, np.array([]), augSourceData['augmented_labels'], [], eeg) # accuracy of augmented data only

            train_acc_list.append(model.train_acc)
            val_acc_list.append(model.val_acc)
            with open(session_directory+"\\model.pkl", 'wb') as file: #save model
                pickle.dump(model, file)

        print()
        for i in range(len(in_fn_list)):
            print(os.path.dirname(in_fn_list[i]) + ' :    train {0:0.2f}      validation {1:0.2f}'.format(train_acc_list[i], val_acc_list[i]))
        print('train acc mean: {0:0.2f}, validation acc mean: {1:0.2f}'.format(np.mean(train_acc_list), np.mean(val_acc_list)))


    else:
        return None

    return model


def train_model_save_source(trials, labels, eeg, session_directory):
    model = MLModel(model_type='csp_lda')
    source_data, labels, pred_labels = model.full_offline_training(trials=trials, labels=labels, eeg=eeg)
    source_data_mat = {'trials':np.transpose(source_data,(0,2,1)), 'labels':labels, 'pred_labels':pred_labels}
    savemat(session_directory + "\\source_data.mat", source_data_mat)
    return model


def load_augmented_data(trials_mat_fp):
    #trials returned as a list of dataframes

    # tr = pickle.load(open(session_directory + "1 Daniel50\\trials.pickle", 'rb'))
    # ch_names = tr[0].columns

    # trials_mat_fp = session_directory + "1 Daniel50\\augmented_train_data_3cond.mat"
    # trials_mat_fp = filedialog.askopenfilename(title='Select trials file', initialdir=session_directory, filetypes=[("mat files", "*.mat")])
    recorded_trials = loadmat(trials_mat_fp)

    trials = []
    labels = []
    augmented_trials = []
    augmented_labels = []
    # ch_names = eeg.get_board_names()
    if 'trials' in recorded_trials and recorded_trials['trials'].any():
        for iTrl in range(recorded_trials['trials'].shape[0]):
            trials.append(pd.DataFrame(recorded_trials['trials'][iTrl,:,:])) #, columns=ch_names))
        labels = recorded_trials['labels'][0].tolist()
    if 'train_data_trials' in recorded_trials and recorded_trials['train_data_trials'].any():
        for iTrl in range(recorded_trials['train_data_trials'].shape[0]):
            trials.append(pd.DataFrame(recorded_trials['train_data_trials'][iTrl,:,:])) #, columns=ch_names))
        labels = recorded_trials['train_data_labels'][0].tolist()
    if 'augmented_data_trials' in recorded_trials and recorded_trials['augmented_data_trials'].any():
        for iTrl in range(recorded_trials['augmented_data_trials'].shape[0]):
            augmented_trials.append(pd.DataFrame(recorded_trials['augmented_data_trials'][iTrl,:,:])) #, columns=ch_names))
        augmented_labels = recorded_trials['augmented_data_labels'][0].tolist()

    # #keep only one label
    # labl = 0
    # for i in sorted(range(len(labels)), reverse=True):
    #     if labels[i] != labl:
    #         labels.pop(i)
    #         trials.pop(i)
    # for i in sorted(range(len(augmented_labels)), reverse=True):
    #     if augmented_labels[i] != labl:
    #         augmented_labels.pop(i)
    #         augmented_trials.pop(i)

    AugmentedData = {'trials':trials, 'labels':labels, 'augmented_trials':augmented_trials, 'augmented_labels':augmented_labels}
    return AugmentedData


if __name__ == '__main__':
    offline_experiment()
