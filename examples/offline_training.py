
from src.bci4als.ml_model import MLModel
from src.bci4als.experiments.offline import OfflineExperiment

def offline_experiment(eeg, load_recorded_trials_flg):

    exp = OfflineExperiment(eeg=eeg, num_trials=60, trial_length=eeg.epoch_len_sec, full_screen=True, audio=False)
    if load_recorded_trials_flg:
        trials, labels, augmented_trials, augmented_labels = exp.load_recorded_trials()
        # augmented_trials = [] #don't use augmentation
        # augmented_labels = []
        # trials = [] #don't use original trials
        # labels = []
    else:
        trials, labels = exp.run()
        augmented_trials = []
        augmented_labels = []

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

    # Classification
    model = MLModel(trials=trials, labels=labels, augmented_trials=augmented_trials,augmented_labels=augmented_labels)
    model.offline_training(eeg=eeg, model_type='csp_lda')

    return model

if __name__ == '__main__':
    offline_experiment()
