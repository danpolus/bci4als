from typing import List
import mne
import pandas as pd
from src.bci4als.eeg import EEG
import numpy as np
from mne.channels import make_standard_montage
from mne_features.feature_extraction import extract_features
from nptyping import NDArray
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import clone
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from autoreject import AutoReject

import sys
sys.path.append('../../../Drone_Project/')
from droneCtrl import Commands

class MLModel:
    """
    A class used to wrap all the ML model train, partial train and predictions
    """

    def __init__(self, model_name: str = '', model_type: str = 'csp_lda'):

        self.model_name = model_name
        self.model_type = model_type
        self.clf = None

        self.nonEEGchannels = ['X1','X2','X3','TRG','CM','A1','A2']
        self.filt_l_freq = 7
        self.filt_h_freq = 30
        self.max_bad_chan_in_epoch = 1
        self.ar = AutoReject(cv=20, thresh_method='bayesian_optimization', random_state=19, verbose=False)
        self.n_csp_comp = 6
        self.csp = CSP(n_components=self.n_csp_comp, transform_into='csp_space')

        self.train_acc = None
        self.val_acc = None

        mpl.use('TkAgg')
        mne.set_log_level('warning')


    def predict(self, trials: List[pd.DataFrame], eeg: EEG):
        epochs = self.convert2mne(trials, eeg)
        if self.model_type.lower() == 'csp_lda':
            epochs, bad_epochs = self.preprocess(epochs, False)
            data = self.csp.transform(epochs.get_data())
            data = extract_features(data, eeg.sfreq, ['pow_freq_bands'], funcs_params={'pow_freq_bands__freq_bands': np.array([self.filt_l_freq,self.filt_h_freq]), 'pow_freq_bands__log': True})
        else:
            raise NotImplementedError('The model type is not implemented yet')

        # Predict
        pred = self.clf.predict(data)
        pred_prob = self.clf.predict_proba(data)
        pred_prob[bad_epochs,:] = 0
        return pred, pred_prob


    def online_predict(self, data: NDArray, eeg: EEG):
        trial = [pd.DataFrame(data.astype(np.float64))]
        pred, pred_prob = self.predict(trial,eeg)

        ##self.enum_image = {0: 'right', 1: 'left', 2: 'idle', 3: 'tongue', 4: 'legs'}
        com_pred = Commands.error
        if pred == 0:
            com_pred = Commands.right
        elif pred == 1:
            com_pred = Commands.left
        elif pred == 2:
            com_pred = Commands.idle
        elif pred == 3:
            com_pred = Commands.forward #tongue
        elif pred == 4:
            com_pred = Commands.back #legs
        return com_pred, pred_prob[0,:].max()


    def full_offline_training(self, trials: List[pd.DataFrame], labels: List[int], eeg: EEG):
        epochs = self.convert2mne(trials, eeg)
        if self.model_type.lower() == 'csp_lda':
            epochs, bad_epochs = self.preprocess(epochs, True)
            epochs = epochs.drop(bad_epochs, verbose=False)
            labels = np.array(labels)[bad_epochs == False].tolist()
            source_data = self.csp.fit_transform(epochs.get_data(), labels)
        else:
            raise NotImplementedError('The model type is not implemented yet')
        pred_labels = self.class_train(source_data, np.array([]), labels, [], eeg)
        if self.model_type.lower() == 'csp_lda':
            return source_data, labels, pred_labels


    def convert2mne(self, trials: List[pd.DataFrame], eeg: EEG):
        #create mne frame
        ch_names = eeg.get_board_names()
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names, eeg.sfreq, ch_types, verbose=False)

        # convert data to mne.Epochs
        trials: List[NDArray] = [t.to_numpy().T for t in trials]
        epochs_array: np.ndarray = np.stack(trials)
        epochs = mne.EpochsArray(epochs_array, info, verbose=False)

        # set montage
        montage = make_standard_montage('standard_1020')
        epochs.drop_channels(self.nonEEGchannels)
        epochs.set_montage(montage,verbose=False)
        # epochs.plot(scalings = dict(eeg=3e1))

        return epochs


    def preprocess(self, epochs, fit_ar_flg):
        epochs.filter(l_freq=self.filt_l_freq, h_freq=self.filt_h_freq, skip_by_annotation='edge', pad='edge', verbose=False) #band-pass filter
        if fit_ar_flg:
            self.ar.fit(epochs)
        reject_log = self.ar.get_reject_log(epochs)
        #reject_log.plot_epochs(epochs,scalings = dict(eeg=3e1))
        bad_epochs = np.logical_or(reject_log.bad_epochs, np.sum(reject_log.labels == 1,axis=1) > self.max_bad_chan_in_epoch)
        # bad_epochs = reject_log.bad_epochs
        # epochs = self.ar.transform(epochs)
        return epochs, bad_epochs


    def class_train(self, data, augmented_data, labels, augmented_labels, eeg: EEG):

        if self.model_type.lower() == 'csp_lda':
            self.clf = LinearDiscriminantAnalysis()
            source_data: np.ndarray = data
            if augmented_data.any():
                # augmented_data = mne.filter.filter_data(augmented_data, l_freq=self.filt_l_freq, h_freq=self.filt_h_freq, sfreq=eeg.sfreq, pad='edge', verbose=False)
                source_data = np.concatenate((source_data,augmented_data))
            trials_data = extract_features(source_data, eeg.sfreq, ['pow_freq_bands'], funcs_params={'pow_freq_bands__freq_bands': np.array([self.filt_l_freq,self.filt_h_freq]), 'pow_freq_bands__log': True})
        else:
            raise NotImplementedError('The model type is not implemented yet')

        #cross validation
        # from sklearn.model_selection import cross_validate, KFold
        # cv_results = cross_validate(self.clf, trials_data, labels, return_train_score=True, cv=KFold(n_splits=5, shuffle=True))
        self.cross_valid(trials_data, labels, augmented_labels, False, False) #show confusion matrices
        crossv_res = {'cv_train':[],'cv_val':[]}
        for i in range(20):
            train_acc, val_acc = self.cross_valid(trials_data, labels, augmented_labels)
            crossv_res['cv_train'] += [train_acc]
            crossv_res['cv_val'] += [val_acc]
        self.train_acc = np.mean(crossv_res['cv_train'])
        self.val_acc = np.mean(crossv_res['cv_val'])
        print('multiple CV:   train acc: {0:0.2f}+-{1:0.3f}, val acc: {2:0.2f}+-{3:0.3f}'.format(self.train_acc, np.std(crossv_res['cv_train']),self.val_acc, np.std(crossv_res['cv_val'])))

        # fit transformer and classifier to data
        self.clf.fit(trials_data, labels + augmented_labels)
        pred_labels = self.clf.predict(trials_data)
        train_acc = metrics.balanced_accuracy_score(labels + augmented_labels, pred_labels)
        print('train acc all trials: {0:0.2f}'.format(train_acc))
        return pred_labels


    def cross_valid(self, all_trials, labels, augmented_labels, plot_flg=False, verbose_flg=False):

        mpl.use('TkAgg')

        nFold = 5
        nTrials = len(labels)
        foldSize = int(np.ceil(nTrials/nFold))

        all_labels = np.array(labels+augmented_labels)

        trials_inx = np.arange(nTrials)
        augmented_trials_inx = np.arange(len(augmented_labels))+nTrials
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

        train_acc = metrics.balanced_accuracy_score(y_train_join, pred_train_join)
        val_acc = metrics.balanced_accuracy_score(y_val_join, pred_val_join)
        if verbose_flg:
            print('train accuracy score: {0:0.2f}'.format(train_acc))
            print('validation accuracy score: {0:0.2f}'.format(val_acc))
        if plot_flg:
            cm_train = metrics.confusion_matrix(y_train_join, pred_train_join)
            cm_val = metrics.confusion_matrix(y_val_join, pred_val_join)
            disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_train)#, display_labels=['right', 'left', 'idle','tongue', 'legs'])
            disp.plot()
            plt.show(block=False)
            disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_val)#, display_labels=['right', 'left', 'idle','tongue', 'legs'])
            disp.plot()
            plt.show(block=False)

        return train_acc, val_acc


    # def svm_train(self, epochs, labels):
    #
    #     from sklearn.pipeline import Pipeline
    #
    #     #preprocessing
    #     epochs.filter(l_freq=self.filt_l_freq, h_freq=self.filt_h_freq, skip_by_annotation='edge', pad='edge', verbose=False) #band-pass filter
    #     mne.set_eeg_reference(epochs, copy=False) #average reference (works badly in some cases, like CSP)   also: epochs = epochs.set_eeg_reference()
    #     epochs = mne.preprocessing.compute_current_source_density(epochs, n_legendre_terms=20) #laplacian works badly
    #     from mne.preprocessing import ICA
    #     ica = ICA(n_components=0.95)
    #     ica.fit(epochs)
    #     ica.plot_components()
    #     ica.plot_sources(epochs)
    #     ecg_indices, _ = ica.find_bads_ecg(epochs, l_freq=self.filt_l_freq)
    #     eog_indices, _ = ica.find_bads_eog(epochs, threshold='auto')
    #     epochs = ica.apply(epochs, exclude=[ecg_indices + eog_indices], n_pca_components=0.95)
    #     #corrmap?  https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
    #
    #     #visualize features
    #     from mne.time_frequency import tfr_morlet
    #     from mne import viz
    #     from mne_features import feature_extraction
    #     # plt.figure()
    #     # axis = plt.axes()
    #     fig = epochs.plot_psd(xscale='log', n_jobs=1, average=False)
    #     # fig.show()
    #     # plt.show(block=False)
    #     freqs = np.logspace(*np.log10([1, 40]), num=20)
    #     power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=freqs/2, use_fft=True, return_itc=True, n_jobs=1)
    #     power.plot_topo(dB=True, title='Average power')
    #     itc.plot_topo(title='Inter-Trial coherence', vmin=0., vmax=1., cmap='Reds')
    #     other_featurs = feature_extraction.extract_features(epochs.get_data(), sfreq, ['std','samp_entropy'])
    #     other_featurs = np.reshape(np.mean(other_featurs, axis=0),[2,18])
    #     _, axis = plt.subplots(1, 4, figsize=(7, 4))
    #     viz.plot_topomap(other_featurs[0,:], epochs.info, vmin=np.min(other_featurs[0,:]), vmax=np.max(other_featurs[0,:]), names = epochs.info.ch_names, show_names=True, cmap='Purples', axes=axis[0]) #std
    #     axis[0].set_title('std')
    #     clim = {'kind': 'value', 'lims': (np.min(other_featurs[0,:]), (np.min(other_featurs[0,:])+np.max(other_featurs[0,:]))/2, np.max(other_featurs[0,:]))}
    #     viz.plot_brain_colorbar(axis[1], clim=clim, colormap='Purples')
    #     viz.plot_topomap(other_featurs[1,:], epochs.info, vmin=np.min(other_featurs[1,:]), vmax=np.max(other_featurs[1,:]), names = epochs.info.ch_names, show_names=True, cmap='Purples', axes=axis[2]) #samp_entropy
    #     axis[2].set_title('samp_entropy')
    #     clim = {'kind': 'value','lims': (np.min(other_featurs[1,:]), (np.min(other_featurs[1,:])+np.max(other_featurs[1,:]))/2, np.max(other_featurs[1,:]))}
    #     viz.plot_brain_colorbar(axis[3], clim=clim, colormap='Purples')
    #     plt.show(block=False)
    #     #https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html
    #
    #
    #     #CSP + LDA classifier
    #     lda = LinearDiscriminantAnalysis()
    #     csp = CSP(n_components=self.n_csp_comp, reg=None, log=True, norm_trace=False)
    #     self.clf = Pipeline([('CSP', csp), ('LDA', lda)])  # Use scikit-learn Pipeline
    #     trials_data = epochs.get_data()
    #     #doi:10.1109/MSP.2008.4408441
    #
    #
    #     #many features + SVM classifier
    #     from sklearn.svm import SVC
    #     from sklearn.preprocessing import StandardScaler
    #     from sklearn.pipeline import make_pipeline
    #     from sklearn.decomposition import PCA
    #     from sklearn.manifold import TSNE
    #     from mne.decoding import CSP
    #     # self.clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear')) #regularizarion C >1. greater C for better generalization
    #     self.clf = SVC(C=1, kernel='linear')
    #     #svm descision boundary visualization: https://scikit-learn.org/0.18/auto_examples/svm/plot_iris.html
    #
    #     trials_data = extract_features(epochs.get_data(), sfreq, ['pow_freq_bands'], funcs_params={'pow_freq_bands__freq_bands': np.array([self.filt_l_freq,self.filt_h_freq]), 'pow_freq_bands__log': True})
    #     trials_data = np.append(trials_data, extract_features(epochs.get_data(), sfreq, ['pow_freq_bands'], funcs_params={'pow_freq_bands__freq_bands': np.arange(self.filt_l_freq,self.filt_h_freq,4), 'pow_freq_bands__log': True}), axis=1)
    #     csp = CSP(n_components=self.n_csp_comp, transform_into='csp_space')
    #     source_data = csp.fit_transform(epochs.get_data(), labels+augmented_labels)
    #     # csp.plot_patterns(epochs.info, ch_type='eeg', show_names=True, units='Patterns (AU)', size=1.5)
    #     # csp.plot_filters(epochs.info, ch_type='eeg', show_names=True, units='Patterns (AU)', size=1.5)
    #     trials_data = extract_features(source_data, sfreq, ['pow_freq_bands'], funcs_params={'pow_freq_bands__freq_bands': np.array([self.filt_l_freq,self.filt_h_freq]), 'pow_freq_bands__log': True})
    #     trials_data = np.append(trials_data, extract_features(source_data, sfreq, ['pow_freq_bands'], funcs_params={'pow_freq_bands__freq_bands': np.arange(self.filt_l_freq,self.filt_h_freq,4), 'pow_freq_bands__log': True}), axis=1)
    #
    #     yt = labels+augmented_labels
    #     pca = PCA()
    #     scaler = StandardScaler()
    #     trials_data = scaler.fit_transform(trials_data,yt)
    #     Xt = pca.fit_transform(trials_data)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.scatter(Xt[:,0], Xt[:,1], Xt[:,2], c=yt)
    #     ax.set_xlabel("PC1 variance: {var:.2f}".format(var = pca.explained_variance_ratio_[0]))
    #     ax.set_ylabel("PC2 variance: {var:.2f}".format(var = pca.explained_variance_ratio_[1]))
    #     ax.set_zlabel("PC3 variance: {var:.2f}".format(var = pca.explained_variance_ratio_[2]))
    #     plt.show(block=False)
    #     tsne = TSNE(n_components=2, perplexity=50, learning_rate='auto', init='random') #play with perplexity and init, see http://arxiv.org/abs/2006.05331
    #     Xt = tsne.fit_transform(trials_data)
    #     plt.figure()
    #     plt.scatter(Xt[:,0], Xt[:,1], c=yt)
    #     plt.show(block=False)
    #
    #
    #     #raw eeg + multi-layer perceptron
    #     from sklearn.neural_network import MLPClassifier
    #     self.clf = MLPClassifier(hidden_layer_sizes=(int(epochs.times.shape[0]/4), int(epochs.times.shape[0]/8), 40, 40, 20), verbose=True)
    #     trials_data = epochs.get_data().reshape((len(epochs),-1))
