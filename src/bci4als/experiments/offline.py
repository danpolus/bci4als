import os
import pickle
import sys
import time
from tkinter import messagebox
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .experiment import Experiment
from src.bci4als.eeg import EEG
from playsound import playsound
from psychopy import visual, event, core
from psychopy.hardware import keyboard


class OfflineExperiment(Experiment):

    def __init__(self, eeg: EEG, trial_length: float, label_keys = (0, 1, 2, 3, 4),
                 cue_length: float = 1.5, ready_length: float = 1,
                 full_screen: bool = False, audio: bool = False):

        super().__init__(eeg)
        self.experiment_type = "Offline"
        self.window_params: Dict[str, Any] = {}
        self.full_screen: bool = full_screen
        self.audio: bool = audio

        # trial times
        self.cue_length: float = cue_length
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

        self.label_keys = label_keys
        self._init_labels(keys=self.label_keys)

        self.kb = keyboard.Keyboard()

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

    def instruction_msg(self):
        global event
        win = self.window_params['main_window']
        color = self.visual_params['text_color']
        instruction_txt = "Let's start with instructions:\n" \
                          "This is a motor-imagery experiment. You should imagine movements but keep your body still.\n" \
                          "- When you see an arrow pointing to the right please imagine yourself moving your right hand\n" \
                          "- When you see an arrow pointing to the left please imagine yourself moving your left hand\n" \
                          "- When you see a green square, rest and don't imagine any movement\n" \
                          "- IF YOU HAVE TO BLINK OR MOVE A BIT, DO IT BETWEEN TRIALS DURING THE PREPARATION PERIOD. \n \n" \
                          "Please press space to continue."
        inst = visual.TextStim(win, instruction_txt, font='arial', color=color)
        inst.setSize(12)
        # inst.font = 'arial 10'

        inst.draw()
        win.flip()
        while 1:
            keys = self.kb.waitKeys()
            for thisKey in keys:
                if thisKey == 'space':
                    return

    def short_break(self):
        global event
        color = self.visual_params['text_color']
        height = self.visual_params['text_height']
        win = self.window_params['main_window']

        msg = visual.TextStim(win, 'short break \n press SPACE to continue', color=color, height=height)
        msg.draw()
        win.flip()
        while 1:
            keys = self.kb.waitKeys()
            for thisKey in keys:
                if thisKey == 'space':
                    return

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
        next_message = visual.TextStim(win, 'The next stimulus is...', pos=[0, 100], color=color, height=height)
        state_text = 'Trial: {} / {}'.format(trial_index + 1, self.num_trials)
        state_message = visual.TextStim(win, state_text, pos=[0, -250], color=color, height=height)
        cue = visual.ImageStim(win, self.images_path[trial_image], pos=[0, -50], size=[630,360])
        cue.draw()
        next_message.draw()
        state_message.draw()
        if self.audio:
            playsound(self.audio_path[trial_image])
        win.flip()
        time.sleep(np.random.uniform(low=self.cue_length, high=self.cue_length+1))

        # # Show ready & state message
        # state_text = 'Trial: {} / {}'.format(trial_index + 1, self.num_trials)
        # state_message = visual.TextStim(win, state_text, pos=[0, -250], color=color, height=height)
        # ready_message = visual.TextStim(win, 'Ready...', pos=[0, 0], color=color, height=height)
        # ready_message.draw()
        # state_message.draw()
        # win.flip()
        # time.sleep(self.ready_length)

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

        messagebox.showinfo(title='Motor Imagery Training', message='Start running trials...')

        # Init psychopy and screen params
        self._init_window()
        self.instruction_msg()

        print(f"Running {self.num_trials} trials")

        # Run trials
        ch_names = self.eeg.get_board_names()
        trials = []
        for i in range(self.num_trials):
            if i == int(self.num_trials / 2):
                self.short_break()

            # Messages for user
            self._user_messages(i)

            # Show stim on window
            self._show_stimulus(i)

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
