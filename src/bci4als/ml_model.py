from typing import List
import mne
import pandas as pd
from src.bci4als.eeg import EEG
import numpy as np
from mne.channels import make_standard_montage
from nptyping import NDArray
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

        #CSP + LDA classifier
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import Pipeline
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
        self.clf = Pipeline([('CSP', csp), ('LDA', lda)])  # Use scikit-learn Pipeline
        trials_data = epochs.get_data()

        # #many features + SVM classifier
        # from sklearn.svm import SVC
        # from sklearn.preprocessing import StandardScaler
        # from sklearn.pipeline import make_pipeline
        # from sklearn.decomposition import PCA
        # from sklearn.manifold import TSNE
        # from mne_features import feature_extraction
        # from mne.decoding import CSP
        # self.clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear')) #regularizarion C >1. greater C for better generalization
        # # self.clf = SVC(C=1, kernel='linear')
        # yt = self.labels+self.augmented_labels
        # trials_data = feature_extraction.extract_features(epochs.get_data(), sfreq, ['pow_freq_bands','kurtosis','rms','hurst_exp','decorr_time','samp_entropy','spect_entropy','spect_slope','hjorth_mobility','hjorth_complexity','teager_kaiser_energy','phase_lock_val','time_corr','spect_corr'], \
        #             funcs_params={'pow_freq_bands__freq_bands': np.arange(self.filt_l_freq, self.filt_h_freq)})
        #             # funcs_params={'pow_freq_bands__freq_bands': np.array([0.5, 4., 8., 12., 28., 40.])})
        # csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
        # trials_data = np.append(trials_data, csp.fit_transform(epochs.get_data(), yt), axis=1) #on validation use transform instead of fit_transform
        # # csp.plot_patterns(epochs.info, ch_type='eeg', show_names=True, units='Patterns (AU)', size=1.5)
        # # csp.plot_filters(epochs.info, ch_type='eeg', show_names=True, units='Patterns (AU)', size=1.5)
        # pca = PCA()
        # scaler = StandardScaler()
        # trials_data = scaler.fit_transform(trials_data,yt)
        # Xt = pca.fit_transform(trials_data)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(Xt[:,0], Xt[:,1], Xt[:,2], c=yt)
        # ax.set_xlabel("PC1 variance: {var:.2f}".format(var = pca.explained_variance_ratio_[0]))
        # ax.set_ylabel("PC2 variance: {var:.2f}".format(var = pca.explained_variance_ratio_[1]))
        # ax.set_zlabel("PC3 variance: {var:.2f}".format(var = pca.explained_variance_ratio_[2]))
        # plt.show(block=False)
        # tsne = TSNE(n_components=2, perplexity=50, learning_rate='auto', init='random') #play with perplexity and init, see http://arxiv.org/abs/2006.05331
        # Xt = tsne.fit_transform(trials_data)
        # plt.figure()
        # plt.scatter(Xt[:,0], Xt[:,1], c=yt)
        # plt.show(block=False)
        # #svm descision boundary visualization: https://scikit-learn.org/0.18/auto_examples/svm/plot_iris.html

        # #raw eeg + multi-layer perceptron
        # from sklearn.neural_network import MLPClassifier
        # self.clf = MLPClassifier(hidden_layer_sizes=(int(epochs.times.shape[0]/4), int(epochs.times.shape[0]/8), 40, 40, 20), verbose=True)
        # trials_data = epochs.get_data().reshape((len(epochs),-1))

        #cross validation
        # from sklearn.model_selection import cross_validate, KFold
        # cv_results = cross_validate(self.clf, trials_data, self.labels, return_train_score=True, cv=KFold(n_splits=5, shuffle=True))
        self.cross_valid(trials_data, True)
        crossv_res = {'cv_train':[],'cv_test':[]}
        for i in range(20):
            train_acc, val_acc = self.cross_valid(trials_data)
            crossv_res['cv_train'] += [train_acc]
            crossv_res['cv_test'] += [val_acc]
        print('train acc: {0:0.2f}+-{1:0.3f}, val acc: {2:0.2f}+-{3:0.3f}'.format(np.mean(crossv_res['cv_train']), np.std(crossv_res['cv_train']), np.mean(crossv_res['cv_test']), np.std(crossv_res['cv_test'])))

        # fit transformer and classifier to data
        self.clf.fit(trials_data, self.labels + self.augmented_labels)

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

    def cross_valid(self, all_trials, plot_flg=False):

        mpl.use('TkAgg')

        nFold = 5
        nTrials = len(self.labels)
        foldSize = int(np.ceil(nTrials/nFold))

        all_labels = np.array(self.labels+self.augmented_labels)

        trials_inx = np.arange(nTrials)
        augmented_trials_inx = np.arange(len(self.augmented_labels))+nTrials
        np.random.shuffle(trials_inx)

        y_train_join = np.array([],dtype=int)
        y_val_join = np.array([],dtype=int)
        pred_val_join = np.array([],dtype=int)
        pred_train_join = np.array([],dtype=int)
        for iFold in range(nFold):
            validation_inx = trials_inx[range(foldSize*iFold, min(foldSize*(iFold+1), nTrials))]
            train_inx = np.append(np.setdiff1d(trials_inx, validation_inx, assume_unique=True), augmented_trials_inx)
            X_train = all_trials[train_inx]
            X_val = all_trials[validation_inx]
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

        train_acc = metrics.accuracy_score(y_train_join, pred_train_join)
        val_acc = metrics.accuracy_score(y_val_join, pred_val_join)
        print('train accuracy score: {0:0.2f}'.format(train_acc))
        print('validation accuracy score: {0:0.2f}'.format(val_acc))
        if plot_flg:
            cm_train = metrics.confusion_matrix(y_train_join, pred_train_join)
            cm_val = metrics.confusion_matrix(y_val_join, pred_val_join)
            disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=['right', 'left', 'idle','tongue', 'legs'])
            disp.plot()
            plt.show(block=False)
            disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=['right', 'left', 'idle','tongue', 'legs'])
            disp.plot()
            plt.show(block=False)

        return train_acc, val_acc
