
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtGui import QIntValidator
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialogButtonBox


class Dialog(QDialog):

    def __init__(self):
        super(Dialog, self).__init__()

        layout = QVBoxLayout(self)

        self.le = QLineEdit()
        self.le.setValidator(QIntValidator())
        self.le.setMaxLength(2)
        self.le.setAlignment(Qt.AlignRight)
        self.le.setFont(QFont("Arial", 20))

        layout.addWidget(self.le)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def gettext():
        dialog = Dialog()
        result = dialog.exec_()
        return(dialog.le.text(),result==QDialog.Accepted)

