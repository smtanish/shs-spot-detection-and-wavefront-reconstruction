# This file is the main GUI entry point that launches live wavefront inference.
# It wires together the inference pipeline, background threading, optional
from PyQt6 import QtWidgets, QtCore
from infer_unet import (
    infer_manager,
    OUTPUT_DIR,
    save_diagnostic_png,
    diagnostic_queue,
)
from wavefront_live import LiveWavefrontViewer
from inference_worker import InferenceWorker
import threading
# Toggle to enable or disable saving diagnostic displacement images.
SAVE_DIAGNOSTICS = False
# This function runs in a background Python thread and saves diagnostic images.

def diagnostic_worker():
    while True:
        item = diagnostic_queue.get()
        # A None item is used as a sentinel to stop the worker cleanly
        if item is None:
            break
        res, output_dir = item
        try:
            save_diagnostic_png(res, output_dir)
        except Exception as e:
            # Diagnostic failures are non-critical and should not crash the app
            print(f"[diagnostic] failed: {e}")
        finally:
            diagnostic_queue.task_done()

def main():
    # Start the diagnostic saving thread only if diagnostics are enabled.
    # This avoids unnecessary background threads when running in live-only mode.
    if SAVE_DIAGNOSTICS:
        threading.Thread(
            target=diagnostic_worker,
            daemon=True,
        ).start()
    # Create the Qt application (required for all Qt widgets)
    app = QtWidgets.QApplication([])
    # Initialize the live OpenGL wavefront viewer
    viewer = LiveWavefrontViewer(grid_size=200)
    # Create a dedicated Qt thread for inference so the UI remains responsive
    thread = QtCore.QThread()
    # Create the inference worker that runs infer_manager in the background
    worker = InferenceWorker(
        infer_manager,
        r"C:\Users\tanis\Desktop\unetcnnoffline\perfectspots\IP",
        r"C:\Users\tanis\Desktop\unetcnnoffline\perfectspots\IA",
        save_outputs=SAVE_DIAGNOSTICS,
        output_root=OUTPUT_DIR,
        n_zernike=10,
        emit_result=viewer.submit,
    )
    # Move the worker object to the background thread
    worker.moveToThread(thread)
    # Start inference when the thread starts
    thread.started.connect(worker.run)
    # Stop the thread cleanly when inference finishes
    worker.finished.connect(thread.quit)
    # Launch the inference thread
    thread.start()
    # Enter the Qt event loop (blocks until the window is closed)
    app.exec()
    # Signal the diagnostic thread to exit cleanly on application shutdown
    if SAVE_DIAGNOSTICS:
        diagnostic_queue.put(None)

if __name__ == "__main__":
    main()
