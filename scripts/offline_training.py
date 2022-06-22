import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
import os
import pickle
from tkinter import filedialog
from sklearn import metrics

from src.bci4als.ml_model import MLModel
from src.bci4als.experiments.offline import OfflineExperiment
from Session import SessionType

def offline_experiment(eeg, sessType: SessionType, train_trials_percent=100):

    session_directory = "C:\My Files\Work\BGU\Datasets\drone BCI"
    if sessType == SessionType.OfflineTrainCspMI:
        in_fn = "\\train_data.mat"
    elif sessType == SessionType.OfflineTrainLdaMI:
        in_fn = "\\alpha_beta_aug15_augmented_source_data.mat"
        model_name = "\\model" # model_30trials

    #######################################################

    model = None
    if sessType == SessionType.OfflineExpMI:
        exp = OfflineExperiment(eeg=eeg, trial_length=eeg.epoch_len_sec, full_screen=False, audio=False)
        trials, labels = exp.run()
        train_data = {'trials':np.stack(trials), 'labels':labels}
        session_directory = exp.session_directory
        savemat(session_directory + "\\train_data.mat", train_data)
        model = train_save_model_source(trials, labels, eeg, session_directory)
        if train_trials_percent < 100:
            trials, labels = reduce_train_data(train_data, labels, train_trials_percent)
            model = train_save_model_source(trials, labels, eeg, session_directory)

    elif sessType == SessionType.OfflineTrainCspMI or sessType == SessionType.OfflineTrainLdaMI:

        in_fn_list = []
        train_acc_list = []
        val_acc_list = []
        in_dir = filedialog.askdirectory(title='Select trials folder', initialdir=session_directory)
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
                cleanTrainData = load_trials_from_file(fn)
                trials=cleanTrainData['trials']
                labels=cleanTrainData['labels']
                if train_trials_percent < 100:
                    trials, labels = reduce_train_data(trials, labels, train_trials_percent)
                model = train_save_model_source(trials, labels, eeg, session_directory)

            else:
                model = pickle.load(open(session_directory+model_name+".pkl", 'rb')) #load model
                augSourceData = load_trials_from_file(fn)
                source_data = np.stack([t.to_numpy().T for t in augSourceData['trials']])
                augmented_source_data = np.stack([t.to_numpy().T for t in augSourceData['augmented_trials']])
                model.class_train(source_data, augmented_source_data, augSourceData['labels'], augSourceData['augmented_labels'], eeg)
                # model.class_train(augmented_source_data, np.array([]), augSourceData['augmented_labels'], [], eeg) # accuracy of augmented data only
                model.name = "model_"+str(len(source_data))+"trials_"+str(len(augmented_source_data))+"NFTaugmented"
                with open(session_directory+"\\"+model.name+".pkl", 'wb') as file: #save model
                    pickle.dump(model, file)
            train_acc_list.append(model.train_acc)
            val_acc_list.append(model.val_acc)

        print()
        for i in range(len(in_fn_list)):
            print(os.path.dirname(in_fn_list[i]) + ' :    train {0:0.2f}      validation {1:0.2f}'.format(train_acc_list[i], val_acc_list[i]))
        print('train acc mean: {0:0.2f}, validation acc mean: {1:0.2f}'.format(np.mean(train_acc_list), np.mean(val_acc_list)))

    elif sessType == SessionType.TestAccuracy:
        subject_models = load_session_models(session_directory)
        test_mat_fp = filedialog.askopenfilename(title='Select recordings file for testing', initialdir=session_directory, filetypes=[("mat files", "*.mat")])
        TestsData = load_trials_from_file(test_mat_fp)
        present_test_accuracy(subject_models, eeg, TestsData['trials'], TestsData['labels'])

    return model

def train_save_model_source(trials, labels, eeg, session_directory):
    model = MLModel(model_name="model_"+str(len(trials))+"trials", model_type='csp_lda')
    source_data, labels, pred_labels = model.full_offline_training(trials=trials, labels=labels, eeg=eeg)
    source_data_mat = {'trials':np.transpose(source_data,(0,2,1)), 'labels':labels, 'pred_labels':pred_labels}
    savemat(session_directory + "\\source_data_"+str(len(trials))+"trials.mat", source_data_mat)
    with open(session_directory+"\\"+model.name+".pkl", 'wb') as file: #save model
        pickle.dump(model, file)
    return model

def reduce_train_data(trials, labels, train_trials_percent):

    reduced_trials = []
    reduced_labels = []
    for Label in np.unique(labels):
        idx = np.where(labels == Label)[0]
        np.random.shuffle(idx)
        idx = idx[0:int(np.round(idx.shape[0]*train_trials_percent/100))]
        reduced_trials += [trials[i] for i in idx]
        reduced_labels += [labels[i] for i in idx]

    # #keep only one label
    # Label = 0
    # for i in sorted(range(len(labels)), reverse=True):
    #     if labels[i] != Label:
    #         labels.pop(i)
    #         trials.pop(i)

    return reduced_trials, reduced_labels

def load_trials_from_file(trials_mat_fp):
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

    TrialsData = {'trials':trials, 'labels':labels, 'augmented_trials':augmented_trials, 'augmented_labels':augmented_labels}
    return TrialsData

def load_session_models(session_directory):
    subject_models = []
    in_dir = filedialog.askdirectory(title='Select session folder with saved models', initialdir=session_directory)
    in_dir_list = os.listdir(in_dir)
    for fn in in_dir_list:
        if not os.path.isdir(in_dir+"\\"+fn) and fn[0:5]=='model' and fn[-4:]=='.pkl':
            model = pickle.load(open(in_dir+"\\"+fn, 'rb'))
            subject_models.append(model)
    return subject_models

def present_test_accuracy(subject_models, eeg, trials, labels):
    labels = np.array(labels)
    for model in subject_models:
        pred_labels, pred_prob = model.predict(trials, eeg)
        test_acc = metrics.balanced_accuracy_score(labels[np.max(pred_prob,axis=1)>0], pred_labels[np.max(pred_prob,axis=1)>0])
        print(model.name+'  test accuracy: {0:0.2f}'.format(test_acc))
