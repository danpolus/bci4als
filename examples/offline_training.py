import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
import os
import pickle
from tkinter import filedialog

from src.bci4als.ml_model import MLModel
from src.bci4als.experiments.offline import OfflineExperiment
from Session import SessionType

def offline_experiment(eeg, model, sessType: SessionType):

    session_directory = "C:\My Files\Work\BGU\Datasets\drone BCI"
    num_trials = 60

    if sessType == SessionType.OfflineExpMI:
        exp = OfflineExperiment(eeg=eeg, num_trials=num_trials, trial_length=eeg.epoch_len_sec, full_screen=True, audio=False)
        trials, labels = exp.run()
        session_directory = exp.session_directory
        train_data = {'trials':np.stack(trials), 'labels':labels}
        savemat(session_directory + "\\train_data.mat", train_data)
        #train a model just to see the accuracy
        model = MLModel(model_type='csp_lda')
        model.full_offline_training(trials=trials, labels=labels, eeg=eeg)

    elif sessType == SessionType.OfflineTrainCspMI:
        cleanTrainData, session_directory = load_augmented_data(session_directory)
        model = MLModel(model_type='csp_lda')
        epochs = model.convert2mne(cleanTrainData['trials'], eeg)
        source_data = model.csp_train(epochs,cleanTrainData['labels'])
        source_data_mat = {'trials':np.transpose(source_data,(0,2,1)), 'labels':cleanTrainData['labels']}
        savemat(session_directory + "\\source_data.mat", source_data_mat)
        #train a model just to see the accuracy
        model.class_train(source_data, np.array([]), cleanTrainData['labels'], [], eeg)

    elif sessType == SessionType.OfflineTrainLdaMI:
        augSourceData, session_directory = load_augmented_data(session_directory)
        source_data = np.stack([t.to_numpy().T for t in augSourceData['trials']])
        augmented_source_data = np.stack([t.to_numpy().T for t in augSourceData['augmented_trials']])
        model.class_train(source_data, augmented_source_data, augSourceData['labels'], augSourceData['augmented_labels'], eeg)
        # model.class_train(augmented_source_data, np.array([]), augSourceData['augmented_labels'], [], eeg) # accuracy of augmented data only

    else:
        return model

    #save model
    with open(session_directory+"\\model.pkl", 'wb') as file:
        pickle.dump(model, file)

    return model


def load_augmented_data(session_directory):
    #trials returned as a list of dataframes

    # tr = pickle.load(open(session_directory + "1 Daniel50\\trials.pickle", 'rb'))
    # ch_names = tr[0].columns

    # trials_mat_fp = session_directory + "1 Daniel50\\augmented_train_data_3cond.mat"
    trials_mat_fp = filedialog.askopenfilename(title='Select trials file', initialdir=session_directory, filetypes=[("mat files", "*.mat")])
    recorded_trials = loadmat(trials_mat_fp)
    session_directory = os.path.dirname(trials_mat_fp)

    trials = []
    labels = []
    augmented_trials = []
    augmented_labels = []
    # ch_names = eeg.get_board_names()
    if recorded_trials['train_data_trials'].any():
        for iTrl in range(recorded_trials['train_data_trials'].shape[0]):
            trials.append(pd.DataFrame(recorded_trials['train_data_trials'][iTrl,:,:])) #, columns=ch_names))
        labels = recorded_trials['train_data_labels'][0].tolist()
    if recorded_trials['augmented_data_trials'].any():
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
    return AugmentedData, session_directory


if __name__ == '__main__':
    offline_experiment()
