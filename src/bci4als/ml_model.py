from typing import List
import mne
import pandas as pd
from src.bci4als.eeg import EEG
import numpy as np
from mne.channels import make_standard_montage
from mne.decoding import CSP
from nptyping import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import clone

import sys
sys.path.append('../../../Drone_Project/')
from droneCtrl import Commands

class MLModel:
    """
    A class used to wrap all the ML model train, partial train and predictions

    ...

    Attributes
    ----------
    trials : list
        a formatted string to print out what the animal says
    """

    def __init__(self, trials: List[pd.DataFrame], labels: List[int]):

        self.trials: List[NDArray] = [t.to_numpy().T for t in trials]
        self.labels: List[int] = labels
        self.debug = True
        self.clf = None
        self.nonEEGchannels = ['X1','X2','X3','TRG','CM']
        self.filt_l_freq = 7
        self.filt_h_freq = 30

    def offline_training(self, eeg: EEG, model_type: str = 'csp_lda'):

        if model_type.lower() == 'csp_lda':

            self._csp_lda(eeg)

        else:

            raise NotImplementedError(f'The model type `{model_type}` is not implemented yet')

    def _csp_lda(self, eeg: EEG):

        print('Training CSP & LDA model')

        # convert data to mne.Epochs
        ch_names = eeg.get_board_names()
        ch_types = ['eeg'] * len(ch_names)
        sfreq: int = eeg.sfreq
        n_samples: int = min([t.shape[1] for t in self.trials])
        epochs_array: np.ndarray = np.stack([t[:, :n_samples] for t in self.trials])

        info = mne.create_info(ch_names, sfreq, ch_types)
        epochs = mne.EpochsArray(epochs_array, info)

        # set montage
        montage = make_standard_montage('standard_1020')
        epochs.drop_channels(self.nonEEGchannels)
        epochs.set_montage(montage)

        # Apply band-pass filter
        epochs.filter(l_freq=self.filt_l_freq, h_freq=self.filt_h_freq, skip_by_annotation='edge', pad='edge', verbose=False)

        # Assemble a classifier
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)

        # Use scikit-learn Pipeline
        self.clf = Pipeline([('CSP', csp), ('LDA', lda)])

        clf_train = clone(self.clf)
        X_train, X_test, y_train, y_test = train_test_split(epochs.get_data(), self.labels, test_size=0.3)  # random_state=0
        clf_train.fit(X_train, y_train)
        pred_train = clf_train.predict(X_train)
        pred_test = clf_train.predict(X_test)
        print('train accuracy score: {0:0.4f}'.format(metrics.accuracy_score(y_train, pred_train)))
        print('test accuracy score: {0:0.4f}'.format(metrics.accuracy_score(y_test, pred_test)))
        cm_train = metrics.confusion_matrix(y_train, pred_train)
        cm_test = metrics.confusion_matrix(y_test, pred_test)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=['right', 'left', 'idle','tongue', 'legs'])
        disp.plot()
        plt.show()
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['right', 'left', 'idle','tongue', 'legs'])
        disp.plot()
        plt.show()

        # fit transformer and classifier to data
        self.clf.fit(epochs.get_data(), self.labels)

    def online_predict(self, data: NDArray, eeg: EEG):
        # Prepare the data to MNE functions
        data = data.astype(np.float64)

        montage = eeg.get_board_names()
        data = np.delete(data, np.where(np.isin(montage, self.nonEEGchannels)), 0)

        # Filter the data ( band-pass only)
        data = mne.filter.filter_data(data, l_freq=self.filt_l_freq, h_freq=self.filt_h_freq, sfreq=eeg.sfreq, pad='edge', verbose=False)

        # Predict
        pred = self.clf.predict(data[np.newaxis])[0]
        pred_prob = self.clf.predict_proba(data[np.newaxis])[0]

        ##self.enum_image = {0: 'right', 1: 'left', 2: 'idle', 3: 'tongue', 4: 'legs'}
        if pred == 0:
            com_pred = Commands.right
        elif pred == 1:
            com_pred = Commands.left
        elif pred == 3:
            com_pred = Commands.forward #tongue
        elif pred == 4:
            com_pred = Commands.back #legs
        else:
            com_pred = Commands.idle #pred==2
        return com_pred, pred_prob.max()

    def partial_fit(self, eeg, X: NDArray, y: int):

        # Append X to trials
        self.trials.append(X)

        # Append y to labels
        self.labels.append(y)

        # Fit with trials and labels
        self._csp_lda(eeg)

