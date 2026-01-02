from PyQt6 import QtCore

class InferenceWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()

    def __init__(self, infer_fn, *args, **kwargs):
        super().__init__()
        self.infer_fn = infer_fn
        self.args = args
        self.kwargs = kwargs

    @QtCore.pyqtSlot()
    def run(self):
        self.infer_fn(*self.args, **self.kwargs)
        self.finished.emit()
