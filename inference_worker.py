# This class is a thin wrapper that runs inference in a background Qt thread.
# It prevents long-running inference from blocking the UI or visualization.

from PyQt6 import QtCore

class InferenceWorker(QtCore.QObject):
    # Signal emitted when inference finishes
    finished = QtCore.pyqtSignal()

    def __init__(self, infer_fn, *args, **kwargs):
        super().__init__()
        # Store the inference function to be executed
        self.infer_fn = infer_fn
        # Positional arguments passed to the inference function
        self.args = args
        # Keyword arguments passed to the inference function
        self.kwargs = kwargs

    # This slot is called when the worker thread starts
    def run(self):
        # Execute the inference function in the background thread
        self.infer_fn(*self.args, **self.kwargs)
        # Notify the main thread that inference has completed
        self.finished.emit()
