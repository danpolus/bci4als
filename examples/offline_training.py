
from src.bci4als.ml_model import MLModel
from src.bci4als.experiments.offline import OfflineExperiment

def offline_experiment(eeg):

    exp = OfflineExperiment(eeg=eeg, num_trials=20, trial_length=eeg.epoch_len_sec, full_screen=True, audio=False)
    trials, labels = exp.run()

    # Classification
    model = MLModel(trials=trials, labels=labels)
    model.offline_training(eeg=eeg, model_type='csp_lda')

    return model

if __name__ == '__main__':
    offline_experiment()
