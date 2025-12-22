import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QEventLoop
import pyqtgraph.opengl as gl
import matplotlib.cm as cm

from wavefront_modal import remove_piston_tilt


class LiveWavefrontViewer(QtCore.QObject):
    def __init__(self, grid_size=150):
        super().__init__()

        # -------------------------------
        # Qt application
        # -------------------------------
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        # -------------------------------
        # Main container widget
        # -------------------------------
        self.container = QtWidgets.QWidget()
        self.container.setWindowTitle("Live Wavefront")
        self.container.resize(900, 700)

        # -------------------------------
        # OpenGL view
        # -------------------------------
        self.view = gl.GLViewWidget(parent=self.container)
        self.view.setBackgroundColor((255, 255, 255, 255))
        self.view.setCameraPosition(
            distance=3.5,
            elevation=12,
            azimuth=60
        )

        # Fill container
        self.view.setGeometry(0, 0, self.container.width(), self.container.height())

        # -------------------------------
        # HUD label (TRUE screen-space)
        # -------------------------------
        self.hud_label = QtWidgets.QLabel(self.container)
        self.hud_label.setText("PV = -- nm   |   RMS = -- nm")
        self.hud_label.setStyleSheet("""
            QLabel {
                color: black;
                background-color: rgba(255, 255, 255, 210);
                padding: 6px 14px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.hud_label.adjustSize()
        self.hud_label.raise_()

        # -------------------------------
        # Grid (normalized pupil)
        # -------------------------------
        self.x = np.linspace(-1.0, 1.0, grid_size)
        self.y = np.linspace(-1.0, 1.0, grid_size)
        self.nx = grid_size
        self.ny = grid_size

        Xg, Yg = np.meshgrid(self.x, self.y, indexing="xy")
        self.pupil_mask = (Xg**2 + Yg**2) <= 1.0

        # -------------------------------
        # Mesh item (no lighting, no edges)
        # -------------------------------
        self.mesh_item = gl.GLMeshItem(
            meshdata=None,
            smooth=False,
            shader=None,
            drawEdges=False
        )
        self.mesh_item.scale(1.0, 1.0, 0.02)
        self.view.addItem(self.mesh_item)

        # -------------------------------
        # Optional minimal axes
        # -------------------------------
        self.axes = gl.GLAxisItem()
        self.axes.setSize(1.2, 1.2, 0.25)
        self.view.addItem(self.axes)

        # -------------------------------
        # Resize handling
        # -------------------------------
        self.container.installEventFilter(self)
        self._position_elements()

        self.container.show()

    # -------------------------------
    # Resize event filter
    # -------------------------------
    def eventFilter(self, obj, event):
        if obj is self.container and event.type() == QtCore.QEvent.Type.Resize:
            self._position_elements()
        return super().eventFilter(obj, event)

    def _position_elements(self):
        w = self.container.width()
        h = self.container.height()

        self.view.setGeometry(0, 0, w, h)

        margin = 12
        self.hud_label.move(
            (w - self.hud_label.width()) // 2,
            h - self.hud_label.height() - margin
        )

    # -------------------------------
    # Colormap (bright & readable)
    # -------------------------------
    def height_to_color(self, values):
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.zeros((len(values), 4), dtype=np.float32)

        z_lo, z_hi = np.nanpercentile(values[finite], [2, 98])
        values = np.clip(values, z_lo, z_hi)

        znorm = (values - z_lo) / (z_hi - z_lo + 1e-12)
        znorm = znorm ** 0.6

        return cm.get_cmap("viridis")(znorm)

    # -------------------------------
    # Update (LIVE)
    # -------------------------------
    def update(self, W_vis, W_phys, X, Y):
        # ---- Metrics ----
        W_metrics = remove_piston_tilt(W_phys, X, Y)
        finite = np.isfinite(W_metrics)

        if np.any(finite):
            pv_nm = (np.nanmax(W_metrics) - np.nanmin(W_metrics)) * 1e9
            rms_nm = np.sqrt(np.nanmean(W_metrics[finite] ** 2)) * 1e9
            self.hud_label.setText(
                f"PV = {pv_nm:.1f} nm   |   RMS = {rms_nm:.1f} nm"
            )
        else:
            self.hud_label.setText("PV = -- nm   |   RMS = -- nm")

        self.hud_label.adjustSize()
        self._position_elements()
        self.hud_label.raise_()

        # ---- Build mesh ----
        Z = np.nan_to_num(W_vis, nan=0.0).astype(np.float32)

        verts = []
        faces = []
        face_vals = []

        idx_map = -np.ones((self.ny, self.nx), dtype=int)
        vcount = 0

        for j in range(self.ny):
            for i in range(self.nx):
                if self.pupil_mask[j, i]:
                    idx_map[j, i] = vcount
                    verts.append([self.x[i], self.y[j], Z[j, i]])
                    vcount += 1

        verts = np.asarray(verts, dtype=np.float32)

        for j in range(self.ny - 1):
            for i in range(self.nx - 1):
                if (
                    self.pupil_mask[j, i] and
                    self.pupil_mask[j, i+1] and
                    self.pupil_mask[j+1, i+1] and
                    self.pupil_mask[j+1, i]
                ):
                    v0 = idx_map[j, i]
                    v1 = idx_map[j, i+1]
                    v2 = idx_map[j+1, i+1]
                    v3 = idx_map[j+1, i]

                    faces.append([v0, v1, v2])
                    faces.append([v0, v2, v3])

                    zf = np.mean([Z[j, i], Z[j, i+1], Z[j+1, i+1], Z[j+1, i]])
                    face_vals.extend([zf, zf])

        faces = np.asarray(faces, dtype=np.int32)
        face_vals = np.asarray(face_vals)

        colors = self.height_to_color(face_vals)

        mesh = gl.MeshData(vertexes=verts, faces=faces)
        mesh.setFaceColors(colors)
        self.mesh_item.setMeshData(meshdata=mesh)

        QtWidgets.QApplication.processEvents(
            QEventLoop.ProcessEventsFlag.AllEvents, 1
        )
