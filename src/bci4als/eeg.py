
from typing import List
from nptyping import NDArray

class EEG:

    def __init__(self, DSIparser, epoch_len_sec):

        self.DSIparser = DSIparser
        self.epoch_len_sec = epoch_len_sec

        # Other Params
        self.sfreq = -1

    def get_board_data(self) -> NDArray:
        """The method returns the data from board and remove it"""
        self.sfreq = self.DSIparser.fsample
        return self.DSIparser.get_epoch(self.epoch_len_sec)

    def get_board_names(self) -> List[str]:
        """The method returns the board's channels"""
        return self.DSIparser.montage

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

    def off(self):
        self.DSIparser.stop()
