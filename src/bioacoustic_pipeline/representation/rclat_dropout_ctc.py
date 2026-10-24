"""Same encoder and parameters; native dropout only between the two LSTM layers."""

from .rclat_simple_ctc import SimpleCTC


class DropoutCTC(SimpleCTC):
    def __init__(self, probability):
        if probability not in (0.0, 0.1):
            raise ValueError("Only the prospectively specified policies are supported")
        super().__init__()
        self.recurrent.dropout = probability
