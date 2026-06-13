import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QTextEdit, QScrollArea, QGroupBox
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from config import STATISTICS, VIEW_NAMES
from pattern_parser import pattern_parser
from processing import process_file
from plots import plot_raw_vs_filtered, plot_onset_detection, plot_features


class EMGApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG Analysis Tool")
        self.setMinimumSize(1100, 800)
        self.emg = None
        self.features_df = None
        self.nav_buttons = {}
        self._build_ui()
        self.current_fig = None
        self.current_view = None
        self.parsed_info = None

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        left = QVBoxLayout()
        left.setSpacing(6)

        file_group = QGroupBox("File")
        fg = QVBoxLayout(file_group)
        load_btn = QPushButton("Load CSV...")
        load_btn.clicked.connect(self._load_file)
        self.file_label = QLabel("no file loaded")
        self.file_label.setWordWrap(True)
        fg.addWidget(load_btn)
        fg.addWidget(self.file_label)
        left.addWidget(file_group)




        views_group = QGroupBox("Views")
        vg = QVBoxLayout(views_group)
        for key, text in [("raw", "Raw vs Filtered"),
                           ("onset", "Onset Detection"),
                           ("features", "Features")]:
            btn = QPushButton(text)
            btn.setEnabled(False)
            btn.clicked.connect(lambda checked, k=key: self._switch_view(k))
            vg.addWidget(btn)
            self.nav_buttons[key] = btn

        left.addWidget(views_group)




        self.stats_group = QGroupBox("Statistics")
        self.stats_layout = QVBoxLayout(self.stats_group)
        self.stats_layout.addWidget(QLabel("load a file to see statistics"))
        left.addWidget(self.stats_group)

        left.addStretch()
        root_layout.addLayout(left, stretch=0)



        right = QVBoxLayout()
        right.setSpacing(6)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.plot_container = QWidget()
        self.plot_inner = QVBoxLayout(self.plot_container)
        self.plot_inner.setContentsMargins(0, 0, 0, 0)
        self.scroll.setWidget(self.plot_container)


        plot_toolbar = QHBoxLayout()
        plot_toolbar.addStretch()
        self.save_btn = QPushButton("Save plot...")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_plot)
        plot_toolbar.addWidget(self.save_btn)
        right.addLayout(plot_toolbar)

        right.addWidget(self.scroll, stretch=1)


        log_group = QGroupBox("Log")
        lg = QVBoxLayout(log_group)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(80)
        self.log.setFont(self.log.font())
        lg.addWidget(self.log)
        right.addWidget(log_group)

        root_layout.addLayout(right, stretch=1)



    def _update_stats(self, view):
        while self.stats_layout.count():
            item = self.stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

        if self.emg is None or view not in STATISTICS:
            self.stats_layout.addWidget(QLabel("load a file to see statistics"))
            return

        for label, fn in STATISTICS[view]:
            try:
                value = fn(self.emg, self.features_df)
            except Exception:
                value = "—"
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{label}:"))
            val_lbl = QLabel(value)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(val_lbl)
            self.stats_layout.addLayout(row)



    def _log(self, text):
        self.log.append(text)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv)")
        if not path:
            return

        name = path.split("/")[-1]
        self.file_label.setText(name)
        self._log(f"loading: {path}")
        self.save_btn.setEnabled(True)

        try:
            self.emg, contractions, self.features_df, removed = process_file(path)
            self._log(f"detected {len(contractions)} contractions")
            self._log(f"removed {removed} rows of padding, {len(self.emg)} rows remaining")
            self._log("done.")

            info = pattern_parser(path)
            self.file_label.setText(info["filename"])
            self.file_label.setText(
                f"Patient: {info['patient']} "
                f"\nLimb: {info['limb']}  "
                f"\nSignal type: {info['signal_type']}  "
                f"\nType of exercise: {info['exercise']}"

            )
            self.parsed_info = pattern_parser(path)
            for btn in self.nav_buttons.values():
                btn.setEnabled(True)
        except Exception as e:
            self._log(f"error: {e}")

    def _switch_view(self, view):
        self.current_view = view
        self._update_stats(view)
        if view == "raw":
            self._show_plot(plot_raw_vs_filtered(self.emg))
        elif view == "onset":
            self._show_plot(plot_onset_detection(self.emg))
        elif view == "features":
            self._show_plot(plot_features(self.emg, self.features_df))

    def _show_plot(self, fig,fixed_height=None):
        self.current_fig = fig
        while self.plot_inner.count():
            item = self.plot_inner.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        dpi = fig.get_dpi()
        w = int(fig.get_figwidth() * dpi)
        h = int(fig.get_figheight() * dpi)

        canvas = FigureCanvas(fig)
        canvas.setFixedSize(w, h)
        self.plot_container.setFixedSize(w, h)
        self.scroll.setWidgetResizable(False)

        self.plot_inner.addWidget(canvas)
        canvas.draw()

    def _save_plot(self):
        if self.current_fig is None:
            return

        if self.parsed_info and self.current_view:
            info = self.parsed_info

            view_str = VIEW_NAMES.get(self.current_view, self.current_view)

            default_name = (
                f"{info['patient']}_"
                f"{info['limb']}_"
                f"{info['signal_type']}_"
                f"{info['exercise']}_"
                f"{view_str}"

            )
        else:
            default_name = "emg_plot"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save plot", default_name, "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;EPS (*.eps)"
        )
        if not path:
            return

        try:
            self.current_fig.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"saved: {path}")
        except Exception as e:
            self._log(f"error saving: {e}")

def run():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = EMGApp()
    window.show()
    sys.exit(app.exec())