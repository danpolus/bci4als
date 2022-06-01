import os
import pickle
import sys
import time
from tkinter import messagebox, filedialog
from typing import Dict, List, Any
import pandas as pd
from .experiment import Experiment
from src.bci4als.eeg import EEG
from playsound import playsound
from psychopy import visual
from scipy.io import savemat, loadmat
import numpy as np


class OfflineExperiment(Experiment):

    def __init__(self, eeg: EEG, num_trials: int, trial_length: float,
                 next_length: float = 1.2, cue_length: float = 0.25, ready_length: float = 1,
                 full_screen: bool = False, audio: bool = False):

        super().__init__(eeg, num_trials)
        self.experiment_type = "Offline"
        self.window_params: Dict[str, Any] = {}
        self.full_screen: bool = full_screen
        self.audio: bool = audio

        # trial times
        self.cue_length: float = cue_length
        self.next_length: float = next_length
        self.ready_length: float = ready_length
        self.trial_length: float = trial_length

        # paths
        self.subject_directory: str = ''
        self.session_directory: str = ''
        self.images_path: Dict[str, str] = {
            'right': os.path.join(os.path.dirname(__file__), 'images', 'arrow_right.jpeg'),
            'left': os.path.join(os.path.dirname(__file__), 'images', 'arrow_left.jpeg'),
            'idle': os.path.join(os.path.dirname(__file__), 'images', 'square.jpeg'),
            'tongue': os.path.join(os.path.dirname(__file__), 'images', 'tongue.jpeg'),
            'legs': os.path.join(os.path.dirname(__file__), 'images', 'legs.jpeg')}
        self.audio_path: Dict[str, str] = {label: os.path.join(os.path.dirname(__file__), 'audio', f'{label}.mp3')
                                           for label in self.enum_image.values()}
        self.audio_success_path = os.path.join(os.path.dirname(__file__), 'audio', f'success.mp3')
        self.visual_params: Dict[str, Any] = {'text_color': 'white', 'text_height': 48}

        self.label_keys = (0, 1, 2) #MI labels fpr training
        self._init_labels(keys=self.label_keys)

        self.signalArray = None

    def _init_window(self):
        """
        init the psychopy window
        :return: dictionary with the window, left arrow, right arrow and idle.
        """

        # Create the main window
        main_window = visual.Window(monitor='testMonitor', units='pix', color='black', fullscr=self.full_screen)

        # Create right, left and idle stimulus
        right_stim = visual.ImageStim(main_window, image=self.images_path['right'])
        left_stim = visual.ImageStim(main_window, image=self.images_path['left'])
        idle_stim = visual.ImageStim(main_window, image=self.images_path['idle'])
        tongue_stim = visual.ImageStim(main_window, image=self.images_path['tongue'])
        legs_stim = visual.ImageStim(main_window, image=self.images_path['legs'])

        self.window_params = {'main_window': main_window, 'right': right_stim, 'left': left_stim,
                              'idle': idle_stim, 'tongue': tongue_stim, 'legs': legs_stim}

    def _user_messages(self, trial_index):
        """
        Show for the user messages in the following order:
            1. Next message
            2. Cue for the trial condition
            3. Ready & state message
        :param trial_index: the index of the current trial
        :return:
        """

        color = self.visual_params['text_color']
        height = self.visual_params['text_height']
        trial_image = self.enum_image[self.labels[trial_index]]
        win = self.window_params['main_window']

        # Show 'next' message & cue & 'play' sound
        next_message = visual.TextStim(win, 'The next stimulus is...', color=color, height=height)
        cue = visual.ImageStim(win, self.images_path[trial_image])

        next_message.draw()
        next_message.pos = (0, 0.2)
        cue.draw()
        cue.pos = (0, -0.2)
        if self.audio:
            playsound(self.audio_path[trial_image])
        win.flip()
        time.sleep(self.next_length)

        # Show ready & state message
        state_text = 'Trial: {} / {}'.format(trial_index + 1, self.num_trials)
        state_message = visual.TextStim(win, state_text, pos=[0, -250], color=color, height=height)
        ready_message = visual.TextStim(win, 'Ready...', pos=[0, 0], color=color, height=height)
        ready_message.draw()
        state_message.draw()
        win.flip()
        time.sleep(self.ready_length)

    def _show_stimulus(self, trial_index):
        """
        Show the current condition on screen and wait.
        Additionally response to shutdown key.
        :param trial_index: the current trial index
        :return:
        """

        # Params
        win = self.window_params['main_window']
        trial_img = self.enum_image[self.labels[trial_index]]
        audio_path = os.path.join(os.path.dirname(__file__), 'audio', '{}.mp3')

        # Play start sound
        if self.audio:
            playsound(audio_path.format('start'))

        # Draw and push marker
        # self.eeg.insert_marker(status='start', label=self.labels[trial_index], index=trial_index)
        self.window_params[trial_img].draw()
        win.flip()

        self.signalArray = None
        time.sleep(self.trial_length / 2)  # Wait
        while self.signalArray is None:  # epoch samples are not ready yet
            time.sleep(self.trial_length / 2) # Wait
            self.signalArray = self.eeg.get_board_data() #zeros[25,600]
        # self.eeg.insert_marker(status='stop', label=self.labels[trial_index], index=trial_index)

        # Play end sound
        if self.audio:
            playsound(audio_path.format('end'))

        # Halt if escape was pressed
        if 'escape' == self.get_keypress():
            sys.exit(-1)

    def _extract_trials(self) -> List[pd.DataFrame]:
        """
        The method extract from the offline experiment collected EEG data and split it into trials.
        The method export a pickle file to the subject directory with a list with all the trials.
        :return: list of trials where each trial is a pandas DataFrame
        """

        # Wait for a sec to the OpenBCI to get the last marker
        time.sleep(0.5)

        # Extract all the recorded data of the whole experiment
        trials = []
        data = self.eeg.get_board_data()
        ch_names = self.eeg.get_board_names()
        ch_channels = self.eeg.get_board_channels()
        durations, labels = self.eeg.extract_trials(data)

        # Assert the labels
        assert self.labels == labels, 'The labels are not equals to the extracted labels'

        # Append each
        for start, end in durations:
            trial = data[ch_channels, start:end]
            trials.append(pd.DataFrame(data=trial.T, columns=ch_names))

        return trials

    def _export_files(self, trials):
        """
        Export the experiment files (trials & labels)
        :param trials:
        :return:
        """
        # Dump to pickle
        trials_path = os.path.join(self.session_directory, 'trials.pickle')
        print(f"Dumping extracted trials recordings to {trials_path}")
        pickle.dump(trials, open(trials_path, 'wb'))

        # Save the labels as csv file
        labels_path = os.path.join(self.session_directory, 'labels.csv')
        print(f"Saving labels to {labels_path}")
        pd.DataFrame.from_dict({'name': self.labels}).to_csv(labels_path, index=False, header=False)

        trials_mat = np.stack(trials)
        train_data = {'trials':trials_mat, 'labels':self.labels}
        savemat(os.path.join(self.session_directory, 'train_data.mat'), train_data)

    def load_recorded_trials(self):

        # tr = pickle.load(open("C:\\My Files\\Work\\BGU\\Datasets\\drone BCI\\1 Daniel50\\trials.pickle", 'rb'))
        # ch_names = tr[0].columns
        # recorded_trials = loadmat('C:/My Files/Work/BGU/Datasets/drone BCI/1 Daniel50/augmented_train_data_3cond.mat')

        trials_mat_fp = filedialog.askopenfilename(title='Select trials file', initialdir="C:\My Files\Work\BGU\Datasets\drone BCI", filetypes=[("mat files", "*.mat")])
        recorded_trials = loadmat(trials_mat_fp)

        self.eeg.on() #get cap eeg settings
        self.eeg.off()
        ch_names = self.eeg.get_board_names()

        trials = []
        augmented_trials = []
        for iTrl in range(recorded_trials['train_data_trials'].shape[0]):
            trials.append(pd.DataFrame(recorded_trials['train_data_trials'][iTrl,:,:], columns=ch_names))
        for iTrl in range(recorded_trials['augmented_data_trials'].shape[0]):
            augmented_trials.append(pd.DataFrame(recorded_trials['augmented_data_trials'][iTrl,:,:], columns=ch_names))

        labels = recorded_trials['train_data_labels'][0].tolist()
        augmented_labels = recorded_trials['augmented_data_labels'][0].tolist()

        return trials, labels, augmented_trials, augmented_labels

    def run(self):
        # Init the current experiment folder
        self.subject_directory = self._ask_subject_directory()
        self.session_directory = self.create_session_folder(self.subject_directory)

        # Start stream
        # initialize headset
        print("Turning EEG connection ON")
        self.eeg.on()

        # Create experiment's metadata
        self.write_metadata()

        messagebox.showinfo(title='bci4als', message='Start running trials...')

        # Init psychopy and screen params
        self._init_window()

        print(f"Running {self.num_trials} trials")

        # Run trials
        ch_names = self.eeg.get_board_names()
        trials = []
        for i in range(self.num_trials):
            # Messages for user
            self._user_messages(i)

            # Show stim on window
            self._show_stimulus(i)

            #save new signalArray
            trials.append(pd.DataFrame(data=self.signalArray.T, columns=ch_names))

        # # Export and return the data
        # trials = self._extract_trials()

        self.window_params['main_window'].close()

        print("Turning EEG connection OFF")
        self.eeg.off()

        # Dump files to pickle
        self._export_files(trials)

        return trials, self.labels
