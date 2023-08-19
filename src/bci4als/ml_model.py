from typing import List
from nptyping import NDArray
import pandas as pd
import numpy as np
import mne
from mne import channels, decoding, preprocessing
from mne_features.feature_extraction import extract_features
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import clone, metrics, discriminant_analysis, model_selection
from autoreject import AutoReject
import sys
sys.path.append('../../../Drone_Project/')

from projectParams import getParams, DroneCommands
from src.bci4als.eeg import EEG

class MLModel:
    """
    A class used to wrap all the ML model train, partial train and predictions
    """

    def __init__(self, model_name: str = ''):

        self.projParams = getParams()

        self.name = model_name
        self.ar = None
        self.decomposition = None
        self.clf = None

        self.acc = {'train_av':-1.0, 'train_std':0, 'valid_av':-1.0, 'valid_std':0, 'test':-1.0}

        mpl.use('TkAgg')
        mne.set_log_level('warning')


    #get prediction from a trained model
    def predict(self, trials: List[pd.DataFrame], eeg: EEG):
        epochs = self.convert2mne(trials, eeg)
        epochs, bad_epochs = self.preprocess_pipeline(epochs)
        data = self.decompose(epochs)
        trials_features = self.calc_features(data, eeg)
        #Predict
        pred = self.clf.predict(trials_features)
        pred_prob = self.clf.predict_proba(trials_features)
        pred_prob[bad_epochs,:] = 0
        return pred, pred_prob


    def online_predict(self, data: NDArray, eeg: EEG):
        trial = [pd.DataFrame(np.transpose(data.astype(np.float64)))]
        pred, pred_prob = self.predict(trial,eeg)
        com_pred = DroneCommands.error
        if pred[0] == 0:
            com_pred = DroneCommands.right
        elif pred[0] == 1:
            com_pred = DroneCommands.left
        elif pred[0] == 2:
            com_pred = DroneCommands.idle
        elif pred[0] == 3:
            com_pred = DroneCommands.forward #tongue
        elif pred[0] == 4:
            com_pred = DroneCommands.back #legs
        return com_pred, pred_prob[0,:].max()


    def calc_test_accuracy(self, eeg: EEG, trials, labels):
        pred_labels, pred_prob = self.predict(trials, eeg)
        acc = metrics.balanced_accuracy_score(labels[np.max(pred_prob,axis=1)>0], pred_labels[np.max(pred_prob,axis=1)>0])
        pe = 1/len(self.projParams['MiParams']['label_keys'])
        self.acc['test'] = (acc-pe)/(1-pe) #kappa


    def full_offline_training(self, trials: List[pd.DataFrame], labels: List[int], eeg: EEG):
        Sources_t = []
        for iFold in range(self.projParams['MiParams']['nFold']+1): #last "fold" is for the full data
            Sources_t.append({'train_source_data': np.array([]), 'train_labels': [], 'aug_source_data': np.array([]), 'aug_labels': [],
                              'train_pred_labels': [], 'valid_source_data': np.array([]), 'valid_labels': [], 'valid_pred_labels': []})
        Sources_t = self.split_folds_calc_source(Sources_t, trials, labels, eeg)
        Sources_t = self.train_and_validate(Sources_t, eeg)
        return Sources_t


    #splits input trials to cv folds; extract sources for each fold; train model for source axraction on the whole input data
    def split_folds_calc_source(self, Sources_t, trials: List[pd.DataFrame], labels: List[int], eeg: EEG):
        skf = model_selection.StratifiedKFold(n_splits=self.projParams['MiParams']['nFold'], shuffle=True)
        iFold = 0
        for train_index, valid_index in skf.split(trials, labels):
            if self.projParams['MiParams']['inverseCV']: #inverse cross-validation. train on 1 fold, test on k-1 folds
                tmp = valid_index
                valid_index = train_index
                train_index = tmp
            Sources_t[iFold]['train_source_data'],Sources_t[iFold]['train_labels'] = self.pipeline_sources([trials[i] for i in train_index], [labels[i] for i in train_index], eeg, train_flg=True)
            Sources_t[iFold]['valid_source_data'],Sources_t[iFold]['valid_labels'] = self.pipeline_sources([trials[i] for i in valid_index], [labels[i] for i in valid_index], eeg, train_flg=False)
            iFold += 1
        Sources_t[-1]['train_source_data'],Sources_t[-1]['train_labels'] = self.pipeline_sources(trials, labels, eeg, train_flg=True) #full data sources and training
        return Sources_t


    #calculate train and validation accuracy on cv folds; train model classifier of sources from the whole data
    def train_and_validate(self, Sources_t, eeg: EEG):
        #validtion accuracy
        train_acc_list = [None] * self.projParams['MiParams']['nFold']
        valid_acc_list = [None] * self.projParams['MiParams']['nFold']
        for iFold in range(self.projParams['MiParams']['nFold']):
            train_acc_list[iFold], Sources_t[iFold]['train_pred_labels'] = self.class_pipeline(Sources_t[iFold]['train_source_data'], Sources_t[iFold]['train_labels'], Sources_t[iFold]['aug_source_data'], Sources_t[iFold]['aug_labels'],  eeg, train_flg=True)
            valid_acc_list[iFold], Sources_t[iFold]['valid_pred_labels'] = self.class_pipeline(Sources_t[iFold]['valid_source_data'], Sources_t[iFold]['valid_labels'], np.array([]), [], eeg, train_flg=False)
        self.acc['train_av'] = np.mean(train_acc_list)
        self.acc['train_std'] = np.std(train_acc_list)
        self.acc['valid_av'] = np.mean(valid_acc_list)
        self.acc['valid_std'] = np.std(valid_acc_list)
        print('AVERAGE ACCURACY:   train {0:0.3f}+-{1:0.3f}, validation {2:0.3f}+-{3:0.3f}'.format(self.acc['train_av'], self.acc['train_std'], self.acc['valid_av'], self.acc['valid_std']))

        # #confusion matrix
        # valid_labels = []
        # pred_labels = []
        # for iFold in range(self.projParams['MiParams']['nFold']):
        #     valid_labels += Sources_t[iFold]['valid_labels']
        #     pred_labels += Sources_t[iFold]['valid_pred_labels']
        # cm_val = metrics.confusion_matrix(Sources_t[0]['valid_labels'], Sources_t[0]['valid_pred_labels'])
        # cm_val = cm_val/np.expand_dims(np.sum(cm_val, axis=1), axis=-1) #normalize
        # disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=['left', 'right'])
        # disp.plot()
        # disp.figure_.axes[0].set_title('Augmented Small Set Accuracy', pad=20, fontsize=40)
        # disp.figure_.set_size_inches(13, 10)
        # plt.show(block=False)
        # plt.rcParams.update({'axes.labelsize': 32})
        # plt.rcParams.update({'font.size': 28})
        # plt.savefig('confusion_2a_augment.png')

        #full training
        full_acc, Sources_t[-1]['train_pred_labels'] = self.class_pipeline(Sources_t[-1]['train_source_data'], Sources_t[-1]['train_labels'], Sources_t[-1]['aug_source_data'], Sources_t[-1]['aug_labels'], eeg, train_flg=True)
        print('FULL TRAIN ACCURACY:   train {0:0.3f}'.format(full_acc))
        return Sources_t


    #preprocess, decompose
    def pipeline_sources(self, trials: List[pd.DataFrame], labels: List[int], eeg: EEG, train_flg = False):
        epochs = self.convert2mne(trials, eeg)
        epochs, bad_epochs = self.preprocess_pipeline(epochs, train_flg)
        epochs = epochs.drop(bad_epochs, verbose=False)
        labels = np.array(labels)[bad_epochs == False].tolist()
        source_data = self.decompose(epochs, labels, train_flg)
        return source_data, labels


    def convert2mne(self, trials: List[pd.DataFrame], eeg: EEG):
        #create mne frame
        ch_names = eeg.get_board_names()
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names, eeg.sfreq, ch_types, verbose=False)
        #convert data to mne.Epochs
        trials: List[NDArray] = [t.to_numpy().T for t in trials]
        epochs_array: np.ndarray = np.stack(trials)
        epochs = mne.EpochsArray(epochs_array, info, verbose=False)
        #set montage
        montage = channels.make_standard_montage('standard_1020')
        epochs.drop_channels(self.projParams['EegParams']['nonEEGchannels'])
        epochs.set_montage(montage,verbose=False)
        # epochs.plot(scalings = dict(eeg=3e1))
        # epochs.plot_psd()
        return epochs


    #filter, clean atrifacts
    def preprocess_pipeline(self, epochs, fit_ar_flg=False):
        epochs.filter(l_freq=self.projParams['MiParams']['l_freq'], h_freq=self.projParams['MiParams']['h_freq'], skip_by_annotation='edge', pad='edge', verbose=False) #band-pass filter
        # epochs = mne.preprocessing.compute_current_source_density(epochs, n_legendre_terms=20)  # laplacian works badly. Needed for features other than CSP
        if self.projParams['MiParams']['clean_epochs_ar_flg']:
            if fit_ar_flg:
                self.ar = AutoReject(cv=5, thresh_method='bayesian_optimization', random_state=19, verbose=False)
                self.ar.fit(epochs)
            reject_log = self.ar.get_reject_log(epochs)
            #reject_log.plot_epochs(epochs,scalings = dict(eeg=3e1))
            bad_epochs = np.logical_or(reject_log.bad_epochs, np.sum(reject_log.labels == 1,axis=1) > self.projParams['MiParams']['max_bad_chan_in_epoch'])
            # bad_epochs = reject_log.bad_epochs
            # epochs = self.ar.transform(epochs)
        else:
            bad_epochs = np.zeros(len(epochs), dtype=bool)
        return epochs, bad_epochs


    def decompose(self, epochs, labels=None, fit_flg=False):
        if self.projParams['MiParams']['decomposition'] == 'CSP':
            if fit_flg:
                self.decomposition = decoding.CSP(n_components=self.projParams['MiParams']['n_csp_comp'], transform_into='csp_space')
                self.decomposition.fit(epochs.get_data(), labels)
                # f = self.decomposition.plot_patterns(epochs.info, ch_type='eeg', sensors=False, show_names=False)
                # f.axes[0].set_title('CSP1: Left hand', fontsize=18)
                # f.axes[1].set_title('CSP2: Right hand', fontsize=18)
                # f.axes[2].set_title('CSP3: Left hand', fontsize=18)
                # f.axes[3].set_title('CSP4: Right hand', fontsize=18)
                # f.axes[4].set_title('[AU]', fontsize=15)
                # f.suptitle("MI Common Spatial Patterns", fontsize=40)
                # f.set_size_inches(13, 6)
                # f.savefig('MIcsp.png')
                # self.decomposition.plot_filters(epochs.info, ch_type='eeg', show_names=True, units='Patterns (AU)', size=1.5)
            return self.decomposition.transform(epochs.get_data())
        elif self.projParams['MiParams']['decomposition'] == 'ICA':
            if fit_flg:
                self.decomposition = preprocessing.ICA(n_components=0.99, method='fastica')
                self.decomposition.fit(epochs)
                # self.ica_eog_idx, scores = self.decomposition.find_bads_eog(epochs, reject_by_annotation=False)
                # self.ica_ecg_idx, scores = self.decomposition.find_bads_ecg(epochs, reject_by_annotation=False)
                # self.ica_muscle_idx, scores = self.decomposition.find_bads_muscle(epochs, reject_by_annotation=False)
            # epochs = self.decomposition.apply(epochs, exclude=self.ica_muscle_idx+self.ica_ecg_idx+self.ica_eog_idx)
            return self.decomposition.get_sources(epochs).get_data()
        elif self.projParams['MiParams']['decomposition'] == None:
            return epochs.get_data()


    def calc_features(self, epoched_data, eeg: EEG):

        #mixed
        if self.projParams['MiParams']['feature'] == 'mixed':
            trials_features = extract_features(epoched_data, eeg.sfreq, ['rms', 'line_length', 'svd_fisher_info']) # hjorth_mobility   wavelet_coef_energy
        #band/multiband power
        elif self.projParams['MiParams']['feature'] == 'BandPower':
            trials_features = extract_features(epoched_data, eeg.sfreq, ['pow_freq_bands'], funcs_params={'pow_freq_bands__freq_bands': self.projParams['MiParams']['power_bands'], 'pow_freq_bands__log': True})
        #RMS
        elif self.projParams['MiParams']['feature'] == 'RMS':
            trials_features = extract_features(epoched_data, eeg.sfreq, ['rms']) # ptp_amp
        #spectral features combination
        elif self.projParams['MiParams']['feature'] == 'Spectral':
            trials_features = extract_features(epoched_data, eeg.sfreq, ['pow_freq_bands', 'rms', 'spect_entropy', 'hjorth_mobility_spect'], funcs_params={
                'pow_freq_bands__freq_bands': self.projParams['MiParams']['power_bands'], 'pow_freq_bands__log': True, 'hjorth_mobility_spect__normalize': True})
        #Higuchi
        elif self.projParams['MiParams']['feature'] == 'Higuchi':
            trials_features = extract_features(epoched_data, eeg.sfreq, ['higuchi_fd'])
        #entropy/fractal combination
        elif self.projParams['MiParams']['feature'] == 'Entropy':
            trials_features = extract_features(epoched_data, eeg.sfreq, ['line_length', 'svd_fisher_info']) # app_entropy samp_entropy  higuchi_fd  katz_fd
            # #Renyi Entropy?
            # from entropy import sample_entropy, app_entropy, perm_entropy, katz_fd, higuchi_fd, detrended_fluctuation, lziv_complexity
            # from scipy import stats, signal
            # from statistics import median
            # trials_features_new = np.empty(shape=(epoched_data.shape[0], epoched_data.shape[1]))
            # for iTrial in range(epoched_data.shape[0]):
            #     for iChan in range(epoched_data.shape[1]):
            #         # trials_features_new[iTrial,iChan] = perm_entropy(epoched_data[iTrial, iChan, :], normalize=True)
            #         # trials_features_new[iTrial, iChan] = higuchi_fd(epoched_data[iTrial, iChan, :])  # higuchi_fd detrended_fluctuation #katz_fd app_entropy sample_entropy
            #
            #         data_proc = epoched_data[iTrial, iChan, :] #e2s Daniel
            #         # data_proc= stats.zscore(epoched_data[iTrial, iChan, :]) # e2s Tomer
            #         # data_proc = np.abs(signal.hilbert(epoched_data[iTrial, iChan, :])) #e2s2 Tomer
            #         data_thresholded = data_proc > median(data_proc)
            #         # data_thresholded = np.ediff1d(epoched_data[iTrial, iChan, :], to_begin=-1) >= 0  # https://www.hindawi.com/journals/mpe/2018/8692146/
            #         trials_features_new[iTrial,iChan] = lziv_complexity(data_thresholded, normalize=True)
        #AR
        elif self.projParams['MiParams']['feature'] == 'AR':
            from statsmodels.tsa.ar_model import AutoReg, ar_select_order
            nARCoef = 4
            trials_features_new = np.empty(shape=(epoched_data.shape[0],epoched_data.shape[1]*nARCoef))
            for iTrial in range(epoched_data.shape[0]):
                for iChan in range(epoched_data.shape[1]):
                    AutoRegModel = AutoReg(epoched_data[iTrial, iChan, :], nARCoef-1) # AutoReg ar_select_order
                    AutoRegRes = AutoRegModel.fit()
                    trials_features_new[iTrial, iChan*nARCoef:(iChan+1)*nARCoef] = AutoRegRes.params
        #MVAR
        elif self.projParams['MiParams']['feature'] == 'MVAR':
            from statsmodels.tsa.vector_ar.var_model import VAR
            nARCoef = 4
            trials_features_new = np.empty(shape=(epoched_data.shape[0],epoched_data.shape[1]*(epoched_data.shape[1]*nARCoef+1)))
            for iTrial in range(epoched_data.shape[0]):
                VarModel = VAR(np.transpose(epoched_data[iTrial, :, :]))
                VarRes = VarModel.fit(nARCoef)
                trials_features_new[iTrial,:] = VarRes.params.flatten()
        # #EMD IMF
        # #https://emd.readthedocs.io/en/stable/   https://pypi.org/project/EMD-signal/
        # elif self.projParams['MiParams']['feature'] == 'EMD_IMF':


        # trials_features = np.append(trials_features, trials_features_new, axis=1)

        return trials_features


    # simple features augmentation, by adding gausian noise to their values
    def augment_features_noise(self, trials_features, labels):
        uniq_labels = np.unique(labels)
        n_aug_trials = int(np.ceil(trials_features.shape[0] / uniq_labels.size) * self.projParams['MiParams']['feature_noise_aug_factor'])
        augmented_features = np.empty(shape=[0, trials_features.shape[1]])
        augmented_labels = []
        for label in uniq_labels:
            feturs_mean = np.mean(trials_features[labels == label, :], axis=0)
            feturs_std = np.std(trials_features[labels == label, :], axis=0)
            trials_features_new = np.random.normal(size=(n_aug_trials, trials_features.shape[1]))
            trials_features_new = trials_features_new * feturs_std * (1 + self.projParams['MiParams']['feature_noise_variation_factor']) + feturs_mean
            augmented_features = np.append(augmented_features, trials_features_new, axis=0)
            augmented_labels = augmented_labels + (label * np.ones(n_aug_trials, dtype=int)).tolist()
        return augmented_features, augmented_labels


    # etract features, train classifier, predict labels. Also consider augmented data when training.
    # no traing mode: just extract features and predict
    def class_pipeline(self, train_data, train_labels, aug_train_data, aug_train_labels, eeg: EEG, train_flg):

        #calculate features
        train_features = self.calc_features(train_data, eeg)
        if train_flg:
            aug_train_features = np.array([])
            if self.projParams['MiParams']['feature_noise_aug_factor'] > 0:
                aug_train_features, aug_train_labels = self.augment_features_noise(train_features,train_labels)
            elif aug_train_data.any():
                aug_train_data = mne.filter.filter_data(aug_train_data, l_freq=self.projParams['MiParams']['l_freq'], h_freq=self.projParams['MiParams']['h_freq'], sfreq=eeg.sfreq, pad='edge', verbose=False)  # needed for non-bandpower features
                aug_train_features = self.calc_features(aug_train_data, eeg)
            if aug_train_features.any():
                train_features = np.append(train_features, aug_train_features, axis=0)

            #fit model
            # if self.projParams['MiParams']['classifier'] == 'LDA': # LDA SVM CNN
            self.clf = discriminant_analysis.LinearDiscriminantAnalysis()

            # #local cross validation
            # # cv_results = model_selection.cross_validate(self.clf, train_features, train_labels+aug_train_labels, return_train_score=True, cv=model_selection.StratifiedKFold(n_splits=self.projParams['MiParams']['nFold'], shuffle=True))
            # self.cross_valid(train_features, train_labels, aug_train_labels, False, False) #show confusion matrices
            # crossv_res = {'cv_train':[],'cv_valid':[]}
            # for i in range(self.projParams['MiParams']['nCV']):
            #     train_acc, valid_acc = self.cross_valid(train_features, train_labels, aug_train_labels)
            #     crossv_res['cv_train'] += [train_acc]
            #     crossv_res['cv_valid'] += [valid_acc]
            # self.acc['train_av'] = np.mean(crossv_res['cv_train'])
            # self.acc['train_std'] = np.std(crossv_res['cv_train'])
            # self.acc['valid_av'] = np.mean(crossv_res['cv_valid'])
            # self.acc['valid_std'] = np.std(crossv_res['cv_valid'])
            # print('multiple CV:   train acc: {0:0.2f}+-{1:0.3f}, valid acc: {2:0.2f}+-{3:0.3f}'.format(self.acc['train_av'], self.acc['train_std'], self.acc['valid_av'], self.acc['valid_std']))

            train_labels = train_labels+aug_train_labels
            self.clf.fit(train_features, train_labels)

        #predict train+augmented labels
        pred_labels = self.clf.predict(train_features).tolist()
        acc = metrics.balanced_accuracy_score(train_labels, pred_labels)
        pe = 1/len(self.projParams['MiParams']['label_keys'])
        acc = (acc-pe)/(1-pe) #kappa
        return acc, pred_labels


    # cross-validation only ant the classification stage (has leakage)
    def cross_valid(self, all_trials, labels, augmented_labels, plot_flg=False, verbose_flg=False):

        nTrials = len(labels)
        foldSize = int(np.ceil(nTrials/self.projParams['MiParams']['nFold']))

        all_labels = np.array(labels+augmented_labels)

        trials_inx = np.arange(nTrials)
        augmented_trials_inx = np.arange(len(augmented_labels))+nTrials
        np.random.shuffle(trials_inx)

        y_train_join = np.array([],dtype=int)
        y_val_join = np.array([],dtype=int)
        pred_val_join = np.array([],dtype=int)
        pred_train_join = np.array([],dtype=int)
        for iFold in range(self.projParams['MiParams']['nFold']):
            validation_inx = trials_inx[range(foldSize*iFold, min(foldSize*(iFold+1), nTrials))]
            if len(validation_inx) == 0:
                continue
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

        train_acc = metrics.balanced_accuracy_score(y_train_join, pred_train_join) #not kappa
        val_acc = metrics.balanced_accuracy_score(y_val_join, pred_val_join) #not kappa
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


    #debug
    # def svm_train(self, epochs, labels):
    #
    #     #preprocessing
    #     epochs.filter(l_freq=self.projParams['MiParams']['l_freq'], h_freq=self.projParams['MiParams']['h_freq'], skip_by_annotation='edge', pad='edge', verbose=False) #band-pass filter
    #     mne.set_eeg_reference(epochs, copy=False) #average reference (works badly in some cases, like CSP)   also: epochs = epochs.set_eeg_reference()
    #     epochs = mne.preprocessing.compute_current_source_density(epochs, n_legendre_terms=20) #laplacian works badly
    #     from mne.preprocessing import ICA
    #     ica = ICA(n_components=0.95)
    #     ica.fit(epochs)
    #     ica.plot_components()
    #     ica.plot_sources(epochs)
    #     ecg_indices, _ = ica.find_bads_ecg(epochs, l_freq=self.projParams['MiParams']['l_freq'])
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
    #     #many features + SVM classifier
    #     from sklearn.svm import SVC
    #     from sklearn.preprocessing import StandardScaler
    #     from sklearn.pipeline import make_pipeline, Pipeline
    #     from sklearn.decomposition import PCA
    #     from sklearn.manifold import TSNE
    #     from mne.decoding import CSP
    #     # self.clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear')) #regularizarion C >1. greater C for better generalization
    #     # self.clf = Pipeline([('scaler', StandardScaler), ('svc', SVC)])
    #     self.clf = SVC(C=1, kernel='linear')
    #     #svm descision boundary visualization: https://scikit-learn.org/0.18/auto_examples/svm/plot_iris.html
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
    #     #raw eeg + multi-layer perceptron
    #     from sklearn.neural_network import MLPClassifier
    #     self.clf = MLPClassifier(hidden_layer_sizes=(int(epochs.times.shape[0]/4), int(epochs.times.shape[0]/8), 40, 40, 20), verbose=True)
    #     trials_data = epochs.get_data().reshape((len(epochs),-1))
    #
