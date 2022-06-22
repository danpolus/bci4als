
import numpy as np
from scipy.io import savemat
import pickle
from src.bci4als.experiments.exp_online import ExpOnline
from scripts.offline_training import load_session_models, present_test_accuracy

def online_experiment(eeg):

    session_directory = "C:\My Files\Work\BGU\Datasets\drone BCI"

    subject_models = load_session_models(session_directory)
    exp = ExpOnline(eeg=eeg, subject_models=subject_models, trial_length=eeg.epoch_len_sec, full_screen=False, audio=False)
    trials, labels = exp.run()
    online_test_data = {'trials':np.stack(trials), 'labels':labels}
    session_directory = exp.session_directory
    savemat(session_directory + "\\online_test_data.mat", online_test_data)
    for model in subject_models:
        with open(session_directory+"\\"+model.name+".pkl", 'wb') as file: #save model
            pickle.dump(model, file)
    present_test_accuracy(subject_models, eeg, trials, labels)
