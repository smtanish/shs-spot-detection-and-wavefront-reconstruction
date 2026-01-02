from PyQt6 import QtWidgets, QtCore
from infer_unet import (
    infer_manager,
    OUTPUT_DIR,
    save_diagnostic_png,
    diagnostic_queue,  
)

from wavefront_live import LiveWavefrontViewer
from inference_worker import InferenceWorker

from queue import Queue
import threading

def diagnostic_worker():
    while True:
        item = diagnostic_queue.get()
        if item is None:
            break 

        res, output_dir = item
        try:
            save_diagnostic_png(res, output_dir)
        except Exception as e:
            print(f"[diagnostic] failed: {e}")
        finally:
            diagnostic_queue.task_done()


def main():

    threading.Thread(
        target=diagnostic_worker,
        daemon=True,
    ).start()

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

    diagnostic_queue.put(None)


if __name__ == "__main__":
    main()
