
import numpy as np
from scipy.io import savemat
import pickle
from tkinter import filedialog
from src.bci4als.experiments.exp_online import ExpOnline
from scripts.offline_training import load_session_models, present_test_accuracy
from projectParams import getParams

def online_experiment(eeg):

    projParams = getParams()

    in_dir = filedialog.askdirectory(title='Select session folder with saved models', initialdir=projParams['FilesParams']['datasetsFp'])
    subject_models = load_session_models(in_dir)
    exp = ExpOnline(eeg=eeg, subject_models=subject_models, trial_length=eeg.epoch_len_sec, label_keys=projParams['MiParams']['label_keys'], full_screen=projParams['MiParams']['full_screen'], audio=projParams['MiParams']['audio'])
    trials, labels = exp.run()
    Trials_t = {'test_trials':np.stack(trials), 'test_labels':labels}
    session_directory = exp.session_directory
    savemat(session_directory+"\\"+projParams['FilesParams']['onlineTestDataFn'], {'Trials_t':Trials_t})
    for model in subject_models:
        with open(session_directory+"\\"+model.name+".pkl", 'wb') as file: #save model
            pickle.dump(model, file)
    present_test_accuracy(subject_models, eeg, trials, labels)
