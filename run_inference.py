from PyQt6 import QtWidgets, QtCore
from infer_unet import infer_manager, OUTPUT_DIR
from wavefront_live import LiveWavefrontViewer
from inference_worker import InferenceWorker


def main():
    app = QtWidgets.QApplication([])

    viewer = LiveWavefrontViewer(grid_size=200)

    thread = QtCore.QThread()
    worker = InferenceWorker(
        infer_manager,
        r"C:\Users\tanis\Desktop\unetcnnoffline\actualshs\IP",
        r"C:\Users\tanis\Desktop\unetcnnoffline\actualshs\IA",
        save_outputs=True,
        output_root=OUTPUT_DIR,
        n_zernike=10,
        emit_result=viewer.submit,
    )

    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)

    thread.start()

    app.exec()


if __name__ == "__main__":
    main()
