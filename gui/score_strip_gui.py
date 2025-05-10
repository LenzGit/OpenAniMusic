from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QTextEdit, QDoubleSpinBox,
)

# ---------------------------------------------------------------------------
# Import the processing back‑end.  Assumes whole_scan.py + detect & helpers
# live in the same directory or are on PYTHONPATH.
# ---------------------------------------------------------------------------
import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from whole_scan import DEFAULT_PROCESSING_PARAMS, batch_process
except Exception as exc:  # pragma: no cover
    print("Could not import whole_scan.py – GUI cannot start.\n", exc)
    sys.exit(1)

import logging, traceback, sys, os
logging.basicConfig(filename='error.log', level=logging.ERROR)

def excepthook(exc_type, exc, tb):
    txt = ''.join(traceback.format_exception(exc_type, exc, tb))
    logging.error(txt)
    QMessageBox.critical(None, "Fatal error", txt)   # optional
    sys.__excepthook__(exc_type, exc, tb)

sys.excepthook = excepthook

# ----------------------------------------------------------------------------
#  Worker thread to run batch_process without freezing the GUI
# ----------------------------------------------------------------------------
class ProcessingThread(QThread):
    log_line = Signal(str)
    finished_ok = Signal()
    finished_error = Signal(str)

    def __init__(self, input_dir: str, output_dir: str, params: Dict[str, Any]):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.params = params

    # pipe stdout to GUI log --------------------------------------------------
    def _redirect_stdout(self):
        import contextlib, io

        class Stream(contextlib.AbstractContextManager):
            def __init__(self, outer):
                self.outer = outer
                self.prev = sys.stdout
                self.buf = io.StringIO()

            def __enter__(self):
                sys.stdout = self  # type: ignore
                return self

            def __exit__(self, *exc):
                sys.stdout = self.prev

            def write(self, txt):  # type: ignore
                self.prev.write(txt)
                if txt.rstrip():
                    self.outer.log_line.emit(txt.rstrip())

            def flush(self):  # type: ignore
                self.prev.flush()

        return Stream(self)

    # heavy work --------------------------------------------------------------
    def run(self):  # noqa: D401
        try:
            with self._redirect_stdout():
                batch_process(self.input_dir, self.output_dir, self.params)
        except Exception:
            tb = traceback.format_exc()
            self.log_line.emit(tb)
            self.finished_error.emit(tb)
            return
        self.finished_ok.emit()


# ----------------------------------------------------------------------------
#  Main Window – all GUI widgets
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sheet‑Strip Extractor")
        self.setWindowIcon(QIcon("icon.svg"))
        self.resize(880, 660)

        central = QWidget()
        self.setCentralWidget(central)
        self.root = QVBoxLayout(central)
        self.root.setAlignment(Qt.AlignTop)

        self._build_forms()
        self._build_log()

        self.thread: ProcessingThread | None = None

    # --------------------------------------------------------------------- UI
    def _build_forms(self):
        #  Paths ------------------------------------------------------------
        path_box = QGroupBox("Paths")
        path_form = QFormLayout(path_box)

        self.le_in = QLineEdit();  self.le_out = QLineEdit()
        btn_in  = QPushButton("Browse …"); btn_out = QPushButton("Browse …")
        btn_in.clicked.connect(lambda: self._pick_dir(self.le_in))
        btn_out.clicked.connect(lambda: self._pick_dir(self.le_out))

        h_in = QHBoxLayout();  h_in.addWidget(self.le_in);  h_in.addWidget(btn_in)
        h_out = QHBoxLayout(); h_out.addWidget(self.le_out); h_out.addWidget(btn_out)
        path_form.addRow("Input folder", h_in)
        path_form.addRow("Output folder", h_out)

        #  Essential parameters -------------------------------------------
        ess_box = QGroupBox("Essential parameters")
        ess_form = QFormLayout(ess_box)

        self.sb_h = QSpinBox(minimum=1, maximum=10000)
        self.sb_h.setValue(DEFAULT_PROCESSING_PARAMS["min_line_length_horizontal"])
        ess_form.addRow("min_line_length_horizontal [px]", self.sb_h)

        self.sb_v = QSpinBox(minimum=1, maximum=10000)
        self.sb_v.setValue(DEFAULT_PROCESSING_PARAMS["min_line_length_vertical"])
        ess_form.addRow("min_line_length_vertical   [px]", self.sb_v)

        self.sb_n = QSpinBox(minimum=1, maximum=500)
        self.sb_n.setValue(1)
        self.sb_n.valueChanged.connect(self._rebuild_table)
        ess_form.addRow("Number of staves", self.sb_n)

        #  Same‑cut controls ----------------------------------------------
        self.cb_same = QCheckBox("use same cut for all staves")
        self.sb_same = QSpinBox(minimum=0, maximum=10000)
        self.sb_same.setValue(0)
        self.cb_same.toggled.connect(self._apply_same_cut)
        self.sb_same.valueChanged.connect(self._apply_same_cut)

        same_row = QHBoxLayout(); same_row.addWidget(self.cb_same); same_row.addWidget(self.sb_same)
        ess_form.addRow("Cut value [px]", same_row)

        #  Table for per‑staff cuts ---------------------------------------
        self.grp_table = QGroupBox("cut_left_while_copy per staff")
        vtab = QVBoxLayout(self.grp_table)
        self.table = QTableWidget(0, 1)
        self.table.setHorizontalHeaderLabels(["cut [px]"])
        vtab.addWidget(self.table)
        self._rebuild_table()

        #  Advanced settings ---------------------------------------------
        adv_box = QGroupBox("Advanced settings"); adv_box.setCheckable(True)
        adv_box.setChecked(False)
        adv_inner = QWidget(); adv_form = QFormLayout(adv_inner)
        self.adv_widgets: Dict[str, Any] = {}
        ESSENTIAL_KEYS = {"threshold_value","horizontal_expansion","vertical_expansion",
                          "min_line_length_horizontal","min_line_length_vertical","cut_left_while_copy"}
        for key, val in DEFAULT_PROCESSING_PARAMS.items():
            if key in ESSENTIAL_KEYS:  # skip essentials
                continue
            w: QWidget
            if isinstance(val, bool):
                w = QCheckBox()
                w.setChecked(val)
            elif isinstance(val, float):  # Check if the default value is a float
                w = QDoubleSpinBox()  # Use QDoubleSpinBox for floats
                w.setMinimum(0.0)  # Set appropriate min/max/decimals
                w.setMaximum(10000.0)  # Example max, adjust as needed
                w.setDecimals(2)  # Example: 2 decimal places, adjust
                w.setSingleStep(0.1)  # Example step, adjust
                w.setValue(val)  # Set the float value directly
            elif isinstance(val, int):  # Check if the default value is an int
                w = QSpinBox()  # Use QSpinBox for integers
                w.setMinimum(0)  # Set appropriate min/max
                w.setMaximum(10000)  # Example max, adjust as needed
                w.setValue(val)  # Set the int value
            else:
                # Fallback for other types, or skip/error
                print(f"Warning: Parameter '{key}' has unhandled type: {type(val)}. Skipping widget.")
                continue

            adv_form.addRow(QLabel(key), w)
            self.adv_widgets[key] = w
        scr = QScrollArea(); scr.setWidgetResizable(True); scr.setWidget(adv_inner)
        QVBoxLayout(adv_box).addWidget(scr)

        #  Generate / Progress -------------------------------------------
        self.btn_gen = QPushButton("Generate"); self.btn_gen.clicked.connect(self._on_generate)
        self.pbar = QProgressBar(); self.pbar.setRange(0,0); self.pbar.hide()

        #  Layout root ----------------------------------------------------
        self.root.addWidget(path_box)
        self.root.addWidget(ess_box)
        self.root.addWidget(self.grp_table)
        self.root.addWidget(adv_box)
        self.root.addWidget(self.btn_gen)
        self.root.addWidget(self.pbar)

    def _build_log(self):
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(140)
        self.root.addWidget(self.log)

    # ---------------------------------------------------------------- workers
    def _pick_dir(self, line: QLineEdit):
        p = QFileDialog.getExistingDirectory(self, "Select folder", os.getcwd())
        if p: line.setText(p)

    def _rebuild_table(self):
        n = self.sb_n.value()
        self.table.setRowCount(n)
        for r in range(n):
            it = QTableWidgetItem("0"); it.setFlags(it.flags() | Qt.ItemIsEditable)
            self.table.setItem(r, 0, it)
        self._apply_same_cut()

    def _apply_same_cut(self):
        same = self.cb_same.isChecked()
        val  = self.sb_same.value()
        self.table.setDisabled(same)
        if same:
            for r in range(self.table.rowCount()):
                self.table.item(r,0).setText("0" if r==0 else str(val))

    # ---------------------------------------------------------------- params
    def _collect(self) -> Dict[str, Any] | None:
        miss: List[str] = []
        inp = self.le_in.text().strip(); out = self.le_out.text().strip()
        if not inp: miss.append("Input folder");
        if not out: miss.append("Output folder")

        h = self.sb_h.value(); v = self.sb_v.value()
        if h<=0: miss.append("min_line_length_horizontal > 0")
        if v<=0: miss.append("min_line_length_vertical > 0")

        # build cut list --------------------------------------------------
        cuts: List[int] = []
        if self.cb_same.isChecked():
            val = self.sb_same.value()
            cuts = [0] + [val]*(self.sb_n.value()-1)
        else:
            for r in range(self.table.rowCount()):
                it = self.table.item(r,0)
                if not it or not it.text().strip():
                    miss.append(f"cut_left_while_copy row {r+1}")
                    continue
                try:
                    cuts.append(int(it.text()))
                except ValueError:
                    miss.append(f"cut_left_while_copy row {r+1} must be int")
            if cuts and cuts[0]!=0:
                cuts[0]=0

        if miss:
            QMessageBox.warning(self, "Missing", "\n".join(miss)); return None

        prm = dict(DEFAULT_PROCESSING_PARAMS)
        prm.update({
            "min_line_length_horizontal": h,
            "min_line_length_vertical"  : v,
            "cut_left_while_copy"       : cuts,
        })
        for k,w in self.adv_widgets.items():
            prm[k] = w.isChecked() if isinstance(w,QCheckBox) else w.value()
        return {"input": inp, "output": out, "params": prm}

    # ---------------------------------------------------------------- slots
    def _on_generate(self):
        col = self._collect();
        if not col: return
        Path(col["output"]).mkdir(parents=True, exist_ok=True)
        self.btn_gen.setDisabled(True); self.pbar.show()
        self.thread = ProcessingThread(col["input"], col["output"], col["params"])
        self.thread.log_line.connect(self.log.append)
        self.thread.finished_ok.connect(self._on_ok)
        self.thread.finished_error.connect(self._on_err)
        self.thread.start()

    @Slot()
    def _on_ok(self):
        self.pbar.hide(); self.btn_gen.setDisabled(False)
        QMessageBox.information(self, "Done", "Processing complete ✓")
        self.thread = None

    @Slot(str)
    def _on_err(self, tb: str):
        self.pbar.hide(); self.btn_gen.setDisabled(False)
        QMessageBox.critical(self, "Error", "Processing failed – see log console")
        self.thread = None


# ----------------------------------------------------------------------------
#  Main entry‑point
# ----------------------------------------------------------------------------

def main():  # noqa: D401
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()