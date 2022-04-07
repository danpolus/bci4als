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
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
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

    def __init__(self, trials: List[pd.DataFrame], labels: List[int], augmented_trials: List[pd.DataFrame], augmented_labels: List[int]):

        self.trials: List[NDArray] = [t.to_numpy().T for t in trials]
        self.labels: List[int] = labels
        self.augmented_trials: List[NDArray] = [t.to_numpy().T for t in augmented_trials]
        self.augmented_labels: List[int] = augmented_labels
        self.debug = True
        self.clf = None
        self.nonEEGchannels = ['X1','X2','X3','TRG','CM','A1','A2']
        self.filt_l_freq = 7
        self.filt_h_freq = 30

        mpl.use('TkAgg')

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
        epochs_array: np.ndarray = np.stack(self.trials + self.augmented_trials)

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

        #cross validation
        self.cross_valid(epochs)

        # fit transformer and classifier to data
        self.clf.fit(epochs.get_data(), self.labels + self.augmented_labels)

        csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
        # self.clf.named_steps['CSP'].plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)

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

    def cross_valid(self, epochs):

        mpl.use('TkAgg')

        nFold = 5
        nTrials = len(self.labels)
        foldSize = int(np.ceil(nTrials/nFold))

        all_trials = epochs.get_data()
        all_labels = np.array(self.labels+self.augmented_labels)

        trials_inx = np.arange(nTrials)
        augmented_trials_inx = np.arange(len(self.augmented_labels))+nTrials
        np.random.shuffle(trials_inx)

        y_train_join = np.array([])
        y_val_join = np.array([])
        pred_val_join = np.array([])
        pred_train_join = np.array([])
        for iFold in range(nFold):
            validation_inx = trials_inx[range(foldSize*iFold, min(foldSize*(iFold+1), nTrials))]
            train_inx = np.append(np.setdiff1d(trials_inx, validation_inx, assume_unique=True), augmented_trials_inx)
            X_train = all_trials[train_inx,:,:]
            X_val = all_trials[validation_inx,:,:]
            y_train = all_labels[train_inx]
            y_val = all_labels[validation_inx]

            clf_val = clone(self.clf)
            clf_val.fit(X_train, y_train)
            pred_train = clf_val.predict(X_train)
            pred_val = clf_val.predict(X_val)

            y_train_join = np.append(y_train_join,y_train)
            y_val_join = np.append(y_val_join,y_val)
            pred_train_join = np.append(pred_train_join,pred_train)
            pred_val_join = np.append(pred_val_join,pred_val)

        print('train accuracy score: {0:0.4f}'.format(metrics.accuracy_score(y_train_join, pred_train_join)))
        print('validation accuracy score: {0:0.4f}'.format(metrics.accuracy_score(y_val_join, pred_val_join)))
        cm_train = metrics.confusion_matrix(y_train_join, pred_train_join)
        cm_val = metrics.confusion_matrix(y_val_join, pred_val_join)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=['right', 'left', 'idle','tongue', 'legs'])
        disp.plot()
        plt.show(block=False)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=['right', 'left', 'idle','tongue', 'legs'])
        disp.plot()
        plt.show(block=False)
