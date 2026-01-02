import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph.opengl as gl
import matplotlib.cm as cm

from wavefront_modal import remove_piston_tilt


class LiveWavefrontViewer(QtCore.QObject):
    def __init__(self, grid_size=150, fps=30):
        super().__init__()

        # -------------------------------
        # Main widget
        # -------------------------------
        self.container = QtWidgets.QWidget()
        self.container.setWindowTitle("Live Wavefront")
        self.container.resize(900, 700)

        # -------------------------------
        # OpenGL view
        # -------------------------------
        self.view = gl.GLViewWidget(self.container)
        self.view.setBackgroundColor((255, 255, 255, 255))
        self.view.setCameraPosition(distance=3.5, elevation=12, azimuth=60)
        self.view.setGeometry(0, 0, 900, 700)

        # -------------------------------
        # HUD (PV / RMS)
        # -------------------------------
        self.hud = QtWidgets.QLabel(self.container)
        self.hud.setStyleSheet("""
            QLabel {
                background: rgba(255,255,255,230);
                padding: 6px 10px;
                font-weight: bold;
                font-size: 13px;
                color: black;
            }
        """)
        self.hud.setText("PV = -- nm | RMS = -- nm")
        self.hud.adjustSize()
        self.hud.move(16, 16)
        self.hud.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hud.raise_()
        self.hud.show()
        # -------------------------------
        # HUD (Render time)
        # -------------------------------
        self.hud_render = QtWidgets.QLabel(self.container)
        self.hud_render.setStyleSheet("""
            QLabel {
                background: rgba(255,255,255,200);
                padding: 4px 8px;
                font-size: 11px;
                color: black;
            }
        """)
        self.hud_render.setText("Render: -- ms")
        self.hud_render.adjustSize()
        self.hud_render.move(16, 44)   # just below PV/RMS
        self.hud_render.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.hud_render.show()
        self.hud_render.raise_()

        # -------------------------------
        # Grid (normalized pupil)
        # -------------------------------
        self.x = np.linspace(-1.0, 1.0, grid_size)
        self.y = np.linspace(-1.0, 1.0, grid_size)
        self.nx = self.ny = grid_size

        Xg, Yg = np.meshgrid(self.x, self.y, indexing="xy")
        self.pupil = (Xg**2 + Yg**2) <= 1.0

        # -------------------------------
        # Precompute mesh topology (ONCE)
        # -------------------------------
        verts = []
        faces = []
        self._vertex_map = -np.ones((self.ny, self.nx), dtype=int)

        v = 0
        for j in range(self.ny):
            for i in range(self.nx):
                if self.pupil[j, i]:
                    self._vertex_map[j, i] = v
                    verts.append([self.x[i], self.y[j], 0.0])
                    v += 1

        for j in range(self.ny - 1):
            for i in range(self.nx - 1):
                if (
                    self.pupil[j, i]
                    and self.pupil[j, i + 1]
                    and self.pupil[j + 1, i + 1]
                    and self.pupil[j + 1, i]
                ):
                    v0 = self._vertex_map[j, i]
                    v1 = self._vertex_map[j, i + 1]
                    v2 = self._vertex_map[j + 1, i + 1]
                    v3 = self._vertex_map[j + 1, i]

                    faces.append([v0, v1, v2])
                    faces.append([v0, v2, v3])

        self._verts = np.asarray(verts, dtype=np.float32)
        self._faces = np.asarray(faces, dtype=np.int32)
        self._colors = np.zeros((len(self._verts), 4), dtype=np.float32)

        self._meshdata = gl.MeshData(
            vertexes=self._verts,
            faces=self._faces
        )
        self._meshdata.setVertexColors(self._colors)

        self.mesh_item = gl.GLMeshItem(
            meshdata=self._meshdata,
            drawEdges=False,
            smooth=False
        )
        self.mesh_item.scale(1.0, 1.0, 0.02)
        self.view.addItem(self.mesh_item)

        self.view.addItem(gl.GLAxisItem())

        # -------------------------------
        # Decoupling state
        # -------------------------------
        self._latest_result = None
        self._lock = QtCore.QMutex()

        # -------------------------------
        # Render timer
        # -------------------------------
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._on_render_tick)
        self._timer.start(int(1000 / fps))

        self.container.show()

    # ==================================================
    # Compute thread entry (non-blocking)
    # ==================================================
    def submit(self, res):
        self._lock.lock()
        self._latest_result = res
        self._lock.unlock()

    # ==================================================
    # Render thread (Qt)
    # ==================================================
    def _on_render_tick(self):
        self._lock.lock()
        res = self._latest_result
        self._latest_result = None
        self._lock.unlock()

        if res is None:
            return

        self._render(
            res["wavefront_vis"],
            res["wavefront"],
            res["X"],
            res["Y"],
        )

    # ==================================================
    # Rendering only (FAST, REUSED MESH)
    # ==================================================
    def _render(self, W_vis, W_phys, X, Y):
        import time
        t0 = time.perf_counter()

        # ---------- Metrics (PHYSICAL) ----------
        W = remove_piston_tilt(W_phys, X, Y)
        finite = np.isfinite(W)
        if not np.any(finite):
            return

        pv = (np.nanmax(W) - np.nanmin(W)) * 1e9
        rms = np.sqrt(np.nanmean(W[finite] ** 2)) * 1e9

        self.hud.setText(f"PV = {pv:.1f} nm | RMS = {rms:.1f} nm")
        self.hud.adjustSize()
        self.hud.raise_()

        # ---------- Visualization (VISUAL) ----------
        Z = np.nan_to_num(W_vis).astype(np.float32)
        finite_z = np.isfinite(Z)

        if np.any(finite_z):
            z_lo, z_hi = np.percentile(Z[finite_z], [2, 98])
        else:
            z_lo, z_hi = 0.0, 1.0

        Zn = np.clip((Z - z_lo) / (z_hi - z_lo + 1e-12), 0, 1) ** 0.6

        # ---------- Update vertices + colors ONLY ----------
        for j in range(self.ny):
            for i in range(self.nx):
                vid = self._vertex_map[j, i]
                if vid >= 0:
                    self._verts[vid, 2] = Z[j, i]
                    self._colors[vid] = cm.viridis(Zn[j, i])

        self._meshdata.setVertexes(self._verts, resetNormals=False)
        self._meshdata.setVertexColors(self._colors)
        self.mesh_item.meshDataChanged()
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0

        self.hud_render.setText(f"Render: {dt_ms:.2f} ms")
        self.hud_render.adjustSize()
        self.hud_render.raise_()