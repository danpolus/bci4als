import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
import os
import pickle
from tkinter import filedialog
import csv

from src.bci4als.ml_model import MLModel
from src.bci4als.experiments.offline import OfflineExperiment
from projectParams import getParams, SessionType

def offline_experiment(eeg, sessType: SessionType, train_trials_percent=100):

    projParams = getParams()

    model = None
    if sessType == SessionType.OfflineExpMI:
        exp = OfflineExperiment(eeg=eeg, trial_length=eeg.epoch_len_sec, label_keys=projParams['MiParams']['label_keys'], full_screen=projParams['MiParams']['full_screen'], audio=projParams['MiParams']['audio'])
        trials, labels = exp.run()
        Trials_t = {'train_trials':np.stack(trials), 'train_labels':labels, 'train_pred_labels':[]}
        session_directory = exp.session_directory
        savemat(session_directory+"\\"+projParams['FilesParams']['trainDataFn'], {'Trials_t':Trials_t})
        model = train_save_model_source(trials, labels, eeg, session_directory, projParams)
        if train_trials_percent < 100:
            trials, labels = reduce_train_data(trials, labels, train_trials_percent)
            model = train_save_model_source(trials, labels, eeg, session_directory, projParams)

    elif sessType == SessionType.OfflineTrainCspMI or sessType == SessionType.OfflineTrainLdaMI or sessType == SessionType.TestAccuracy:

        in_fn_list = []
        acc_lists = {'train_acc':[], 'train_acc_std':[], 'valid_acc':[], 'valid_acc_std':[], 'test_acc':[]}
        in_dir = filedialog.askdirectory(title='Select trials folder', initialdir=projParams['FilesParams']['datasetsFp'])
        # in_dir = projParams['FilesParams']['datasetsFp']+"\\2a"

        if sessType == SessionType.OfflineTrainCspMI:
            in_fn = "\\"+projParams['FilesParams']['trainDataFn']
        elif sessType == SessionType.OfflineTrainLdaMI:
            in_fn = "\\"+projParams['FilesParams']['augSourceDataFn']
        elif sessType == SessionType.TestAccuracy:
            in_fn = "\\"+projParams['FilesParams']['testDataFn']

        if os.path.exists(in_dir+in_fn):
            in_fn_list.append(in_dir+in_fn)
        else:
            in_dir_list = os.listdir(in_dir)
            for fp in in_dir_list:
                if os.path.isdir(in_dir+"\\"+fp) and os.path.exists(in_dir+"\\"+fp+in_fn):
                    in_fn_list.append(in_dir+"\\"+fp+in_fn)
        if len(in_fn_list) == 0:
            print('No matching recording files. Please select a specific file')
            fn = filedialog.askopenfilename(title='Select recordings file', initialdir=projParams['FilesParams']['datasetsFp'], filetypes=[("mat files", "*.mat")])
            in_fn_list = [fn]

        for fn in in_fn_list:
            session_directory = os.path.dirname(fn)
            trials_mat = loadmat(fn, simplify_cells=True)

            if sessType == SessionType.OfflineTrainCspMI:
                trials, labels = trials_labels_to_lists(trials_mat['Trials_t']['train_trials'], trials_mat['Trials_t']['train_labels'])
                trials, labels = select_classes(trials, labels, projParams['MiParams']['label_keys'])
                if train_trials_percent < 100:
                    trials, labels = reduce_train_data(trials, labels, train_trials_percent)
                model = train_save_model_source(trials, labels, eeg, session_directory,projParams)

            elif sessType == SessionType.OfflineTrainLdaMI:
                model = pickle.load(open(session_directory+"\\"+projParams['FilesParams']['cspFittedModelName'], 'rb')) #load model
                model.projParams = projParams
                for iFold in range(len(trials_mat['Sources_t'])):
                    trials_mat['Sources_t'][iFold]['train_labels'] = trials_mat['Sources_t'][iFold]['train_labels'].tolist()
                    trials_mat['Sources_t'][iFold]['aug_labels'] = trials_mat['Sources_t'][iFold]['aug_labels'].tolist()
                    trials_mat['Sources_t'][iFold]['valid_labels'] = trials_mat['Sources_t'][iFold]['valid_labels'].tolist()
                model.train_and_validate(trials_mat['Sources_t'], eeg)
                model.name = projParams['MiParams']['input_prefix']+model.name
                # model.train_and_validate(augmented_source_data, augSourceData['augmented_labels'], [], [], valid_source_data, augSourceData['valid_labels'], eeg) # accuracy of augmented data only
                with open(session_directory+"\\"+model.name+".pkl", 'wb') as file: #save model
                    pickle.dump(model, file)

            elif sessType == SessionType.TestAccuracy:
                subject_models = load_session_models(session_directory)
                trials, labels = trials_labels_to_lists(trials_mat['Trials_t']['test_trials'], trials_mat['Trials_t']['test_labels'])
                trials, labels = select_classes(trials, labels, projParams['MiParams']['label_keys'])
                print(fn+" :")
                subject_models = present_test_accuracy(subject_models, eeg, trials, labels)
                model = subject_models[0] #for test accuracy statistics

            acc_lists['train_acc'].append(model.acc['train_av'])
            acc_lists['train_acc_std'].append(model.acc['train_std'])
            acc_lists['valid_acc'].append(model.acc['valid_av'])
            acc_lists['valid_acc_std'].append(model.acc['valid_std'])
            acc_lists['test_acc'].append(model.acc['test'])

        print()
        for i in range(len(in_fn_list)):
            print(os.path.dirname(in_fn_list[i]) + ' :    train {0:0.2f}      validation {1:0.2f}      test {2:0.2f}'.format(acc_lists['train_acc'][i], acc_lists['valid_acc'][i], acc_lists['test_acc'][i]))
        print('AVERAGE ACCURACY:   train {0:0.3f}+-{1:0.3f}, validation {2:0.3f}+-{3:0.3f}, test {4:0.3f}+-{5:0.3f}'.format(np.mean(acc_lists['train_acc']), np.std(acc_lists['train_acc']), np.mean(acc_lists['valid_acc']), np.std(acc_lists['valid_acc']), np.mean(acc_lists['test_acc']), np.std(acc_lists['test_acc'])))

        with open(in_dir+"\\"+projParams['FilesParams']['classResults'],'w') as f:
            writer = csv.writer(f,lineterminator='\r')
            for i in range(len(in_fn_list)):
                writer.writerow([acc_lists['train_acc'][i], acc_lists['valid_acc'][i], acc_lists['test_acc'][i], acc_lists['train_acc_std'][i], acc_lists['valid_acc_std'][i]])

    return model

def train_save_model_source(trials, labels, eeg, session_directory, projParams):
    if projParams['FilesParams']['cspFittedModelName'] != None:
        model = MLModel(model_name=projParams['FilesParams']['cspFittedModelName'][0:-4])
    else:
        model = MLModel(model_name="model_"+str(len(trials))+"trials")
    Sources_t = model.full_offline_training(trials, labels, eeg)
    if projParams['FilesParams']['sourceDataFn'] != None:
        savemat(session_directory+"\\"+projParams['FilesParams']['sourceDataFn'],  {'Sources_t': Sources_t})
    else:
        savemat(session_directory + "\\source_data_"+str(len(trials))+"trials.mat",  {'Sources_t': Sources_t})
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
    return reduced_trials, reduced_labels

def select_classes(trials, labels, label_keys):
    for i in reversed(range(len(labels))):
        if labels[i] not in label_keys:
            labels.pop(i)
            trials.pop(i)
    return trials, labels

def trials_labels_to_lists(trials_nd, labels_nd):
    # ch_names = eeg.get_board_names()
    trials = []
    for iTrl in range(trials_nd.shape[0]):
        trials.append(pd.DataFrame(trials_nd[iTrl,:,:])) #, columns=ch_names))
    labels = labels_nd.tolist()
    return trials, labels

def load_session_models(in_dir):
    subject_models = []
    in_dir_list = os.listdir(in_dir)
    for fn in in_dir_list:
        if not os.path.isdir(in_dir+"\\"+fn) and fn[0:5]=='model' and fn[-4:]=='.pkl':
            model = pickle.load(open(in_dir+"\\"+fn, 'rb'))
            subject_models.append(model)
    return subject_models

def present_test_accuracy(subject_models, eeg, trials, labels):
    labels = np.array(labels)
    for model in subject_models:
        model.calc_test_accuracy(eeg,trials,labels)
        print(model.name+'  test accuracy: {0:0.2f}'.format(model.acc['test']))
    return subject_models
