import os
from io import StringIO

import numpy as np
import shutil


def flush_buffer(buffer, output_file):
    """
    Write specific buffer to corresponding output file.
    :param buffer: Buffer that will be flushed
    :param output_file: File to which the buffer will be flushed
    """
    with open(output_file, 'a') as f:
        buffer.seek(0)
        shutil.copyfileobj(buffer, f, -1)
    # clear buffer
    return StringIO()


class TrainLogger:
    """
    Logger responsible for output files of training statistics
    """

    def __init__(self, output_file, save_freq=1):
        """
        :param output_file: str
            Filename to write results for each iteration.
        :param save_freq: int
            Outputs are flushed to file every save_freq epochs (default == 1).
            If < 0, uses save_freq = inf.
        """
        self.output_file = output_file
        self.save_freq = save_freq

        if save_freq is None:
            self.buffer_frequency = 1
        elif save_freq < 1:
            self.buffer_frequency = float('inf')
        else:
            self.buffer_frequency = save_freq

        self.buffer_epoch_stats = StringIO()  # buffer for epoch statistics

    def setup_output_files(self):
        """
        Opens and prepares all output log files controlled by this class.
        """
        if self.output_file is not None:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w') as f:
                headers = ["epoch",
                           "p_ob",
                           "v_ob",
                           "qu_ob",
                           "qd_ob",
                           "p_ca",
                           "v_ca",
                           "qu_ca",
                           "qd_ca",
                           "tau_0_ca",
                           "tau_omega_ca",
                           "Cbar_ca",
                           "Qbar_ca",
                           "loss_obs",
                           "loss_phy",
                           "loss_hid",
                           "total_loss"]
                f.write("{}\n".format(",".join(headers)))

    def save_stats(self, epoch, loss_obs, loss_phy, loss_hid, loss):
        """
        Computes and saves all outputs computed for every epoch.
        Depending on the value of self.buffer_frequency.
        """
        epoch = epoch + 1  # changing from 0-based index to 1-based
        # we first save any pending buffer
        self.flush_buffers()

        if self.output_file is not None:
            stats = np.array([[
                epoch,
                loss_obs.detach().numpy(),
                loss_phy.detach().numpy(),
                loss_hid.detach().numpy(),
                loss.detach().numpy()
            ]], dtype=np.float32)
            np.savetxt(self.buffer_epoch_stats, stats, delimiter=',')

        # should the buffer be saved now?
        if epoch % self.buffer_frequency == 0:
            self.flush_buffers()

    def flush_buffers(self):
        """
        Write all buffers to file.
        """
        if self.output_file is not None:
            self.buffer_epoch_stats = flush_buffer(
                self.buffer_epoch_stats, self.output_file)
