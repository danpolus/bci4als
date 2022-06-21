
import numpy as np
from scipy.io import savemat
import pickle
from bci4als.experiments.online import OnlineExperiment
from scripts.offline_training import load_session_models, present_test_accuracy

def online_experiment(eeg):

    session_directory = "C:\My Files\Work\BGU\Datasets\drone BCI"

    subject_models = load_session_models(session_directory)

    exp = OnlineExperiment(eeg=eeg, model=subject_models, num_trials=10, buffer_time=4, threshold=3, skip_after=8, co_learning=True, debug=False) # co_learning=False:   predict w/wo training
    trials, labels = exp.run(use_eeg=True, full_screen=True)
    exp_data = {'trials':np.stack(trials), 'labels':labels}
    session_directory = exp.session_directory
    savemat(session_directory + "\\exp_data.mat", exp_data)
    for model in subject_models:
        with open(session_directory+"\\"+model.model_name+".pkl", 'wb') as file: #save model
            pickle.dump(model, file)

    present_test_accuracy(subject_models, eeg, trials, labels)
