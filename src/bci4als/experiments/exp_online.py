
from psychopy import visual, core
import random
import pandas as pd
import numpy as np
from tkinter import messagebox
from typing import List

from .offline import OfflineExperiment
from src.bci4als.eeg import EEG
from src.bci4als.ml_model import MLModel

class ExpOnline(OfflineExperiment):

    def __init__(self, eeg: EEG, subject_models: List[MLModel], trial_length: float, label_keys: tuple,
                 full_screen: bool = False, audio: bool = False):
        super().__init__(eeg, trial_length=trial_length, label_keys=label_keys, full_screen=full_screen, audio=audio)

        self.experiment_type = "ExpOnline"
        self.subject_models = subject_models


    def _show_feedback(self, trial_index, model):
        pred_label, pred_prob = model.predict([pd.DataFrame(self.signalArray.T)], self.eeg)
        if pred_prob.max() > 0:
            if pred_label == self.labels[trial_index]:
                msg = 'Correct!'
            else:
                msg = 'Wrong!  Predicted label is: {}'.format(pred_label[0])
            print(msg)
        else:
            print('bad signal, failed to predict')


    def run(self):
        global visual, core, event
        # init the trials number
        self.ask_num_trials()
        self._init_labels(keys=self.label_keys)

        # Init the current experiment folder
        self.subject_directory = self._ask_subject_directory()
        self.session_directory = self.create_session_folder(self.subject_directory)

        # Start stream
        # initialize headset
        print("Turning EEG connection ON")
        self.eeg.on()

        # Create experiment's metadata
        self.write_metadata()

        message = 'Start running {} trials \n\nModels in use:'.format(self.num_trials)
        for model in self.subject_models:
            message += '\n'+model.name
        messagebox.showinfo(title='Motor Imagery Testing', message=message)

        # Init psychopy and screen params
        self._init_window()

        print(f"Running {self.num_trials} trials")

        model_inx = np.arange(len(self.labels)) % len(self.subject_models)
        random.shuffle(model_inx)

        # Run trials
        ch_names = self.eeg.get_board_names()
        trials = []
        for i in range(self.num_trials):
            # Messages for user
            self._user_messages(i)

            # Show stim on window
            self._show_stimulus(i)

            # Show model prediction to the user
            self._show_feedback(i, self.subject_models[model_inx[i]])

            keys = self.kb.getKeys()
            for thisKey in keys:
                if thisKey == 'escape':
                    core.quit()

            #save new signalArray
            trials.append(pd.DataFrame(data=self.signalArray.T, columns=ch_names))

        self.window_params['main_window'].close()

        print("Turning EEG connection OFF")
        self.eeg.off()

        # Dump files to pickle
        self._export_files(trials)

        return trials, self.labels
