
from typing import List

import numpy as np
from nptyping import NDArray
import time

class EEG:

    def __init__(self, DSIparser, projParams):

        self.DSIparser = DSIparser
        self.epoch_len_sec = projParams['EegParams']['epoch_len_sec']
        self.sfreq = projParams['EegParams']['sfreq']
        self.chan_names = projParams['EegParams']['chan_names']

    def get_board_data(self):# -> ndarray:
        """The method returns the data from board and remove it"""
        if self.DSIparser is None:
            return np.zeros((len(self.chan_names), self.epoch_len_sec*self.sfreq)) #just for debug
        return self.DSIparser.get_epoch(self.epoch_len_sec)

    def get_board_names(self) -> List[str]:
        """The method returns the board's channels"""
        return self.chan_names

    def get_board_channels(self):
        """Get list with the channels locations as list of int"""
        # from mne.channels import make_standard_montage
        # return make_standard_montage('standard_1020')
        return self.DSIparser.montage

    def get_channels_data(self):
        """Get NDArray only with the channels data (without all the markers and other stuff)"""
        return self.get_board_data()

    def clear_board(self):
        """Clear all data from the EEG board"""
        # Get the data and don't save it
        self.DSIparser.get_epoch(self.DSIparser.fifo_len_sec)

    def on(self):
        self.DSIparser.start()
        # time.sleep(0.2)  # wait the thread to start
        # self.sfreq = self.DSIparser.fsample
        # self.chan_names = self.DSIparser.montage
        return

    def off(self):
        self.DSIparser.stop()
        return
