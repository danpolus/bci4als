
from typing import List
from nptyping import NDArray
import time

class EEG:

    def __init__(self, DSIparser, epoch_len_sec):

        self.DSIparser = DSIparser
        self.epoch_len_sec = epoch_len_sec

        # Other Params
        self.sfreq = -1
        self.chan_names = []

    def get_board_data(self) -> NDArray:
        """The method returns the data from board and remove it"""
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
        time.sleep(0.2) #wait the thread to start
        self.sfreq = self.DSIparser.fsample
        self.chan_names = self.DSIparser.montage

    def off(self):
        self.DSIparser.stop()

    # #when running offline, use this version
    # def on(self):
    #     self.sfreq = 300
    #     self.chan_names = ['P3','C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz','CM', 'A1', 'Fp1', 'Fp2' , 'T3', 'T5', 'O1', 'O2', 'X3' , 'X2', 'F7', 'F8', 'X1', 'A2', 'T6', 'T4', 'TRG']
    #
    # def off(self):
    #     return
