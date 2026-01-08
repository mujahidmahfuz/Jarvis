"""
Global stylesheet for the Pocket AI application.
"""

STYLESHEET = """
/* Main Window */
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #121212, stop:1 #1e1e24);
}

/* Sidebar */
QFrame#sidebar {
    background-color: rgba(30, 30, 35, 180);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

QLabel#sidebarTitle {
    color: #4F8EF7;
    font-size: 20px;
    font-weight: bold;
    padding: 15px;
    letter-spacing: 3px;
}

QPushButton#newChatBtn {
    background-color: rgba(255, 255, 255, 0.05);
    color: #e8eaed;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 12px 15px;
    font-size: 14px;
    font-weight: 500;
    text-align: left;
}

QPushButton#newChatBtn:hover {
    background-color: rgba(255, 255, 255, 0.1);
    border: 1px solid #4F8EF7;
    color: white;
}

QPushButton#newChatBtn:pressed {
    background-color: rgba(255, 255, 255, 0.15);
}

QListWidget#sessionList {
    background-color: transparent;
    border: none;
    outline: none;
    padding: 5px 10px;
}

QListWidget#sessionList::item {
    background-color: transparent;
    color: #9e9e9e;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    min-height: 36px;
}

QListWidget#sessionList::item:hover {
    background-color: rgba(255, 255, 255, 0.05);
    color: #e8eaed;
}

QListWidget#sessionList::item:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(79, 142, 247, 0.2), stop:1 rgba(79, 142, 247, 0.05));
    color: white;
    border-left: 2px solid #4F8EF7;
}

/* Chat Panel */
QFrame#chatPanel {
    background-color: transparent;
}

QFrame#chatHeader {
    background-color: rgba(30, 30, 35, 180);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

QLabel#headerTitle {
    color: #e8eaed;
    font-size: 16px;
    font-weight: 500;
}

QLabel#statusLabel {
    color: #6e6e6e;
    font-size: 12px;
}

/* Scroll Area */
QScrollArea#chatScroll {
    background-color: transparent;
    border: none;
}

QScrollArea#chatScroll > QWidget > QWidget {
    background-color: transparent;
}

QScrollBar:vertical {
    background: transparent;
    width: 6px;
    border-radius: 3px;
}

QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 3px;
    min-height: 40px;
}

QScrollBar::handle:vertical:hover {
    background: rgba(255, 255, 255, 0.3);
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* Input Area */
QFrame#inputBar {
    background-color: rgba(30, 30, 35, 180);
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}

QLineEdit#userInput {
    background-color: rgba(255, 255, 255, 0.05);
    color: #e8eaed;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 12px 20px;
    font-size: 14px;
}

QLineEdit#userInput:focus {
    background-color: rgba(255, 255, 255, 0.08);
    border: 1px solid #4F8EF7;
}

QPushButton#sendBtn, QPushButton#stopBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4F8EF7, stop:1 #bd93f9);
    border: none;
    border-radius: 22px; /* Half of 44px */
    min-width: 44px;
    max-width: 44px;
    min-height: 44px;
    max-height: 44px;
    font-size: 16px;
}

QPushButton#sendBtn {
    color: white;
}

QPushButton#stopBtn {
    background-color: #ef5350; /* Keep red for stop */
    background: #ef5350;
    color: white;
}

QPushButton#sendBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3d7be5, stop:1 #a87df0);
}

QPushButton#sendBtn:pressed {
    background: #333;
}

/* TTS Toggle Switch */
QCheckBox#ttsToggle {
    color: #9e9e9e;
    font-size: 12px;
    spacing: 8px;
}

QCheckBox#ttsToggle::indicator {
    width: 44px;
    height: 24px;
    border-radius: 12px;
    background-color: #3d3d3d;
    border: 2px solid #3d3d3d;
}

QCheckBox#ttsToggle::indicator:checked {
    background-color: #4F8EF7;
    border: 2px solid #4F8EF7;
    image: none;
}

/* Delete button in session list */
QPushButton#deleteBtn {
    background-color: transparent;
    color: #6e6e6e;
    border: none;
    font-size: 14px;
    padding: 5px;
}

QPushButton#deleteBtn:hover {
    color: #ef5350;
}

/* Top Tab Bar */
QTabBar#topTabBar {
    background: transparent;
    border: none;
}

QTabBar#topTabBar::tab {
    background: transparent;
    color: #9e9e9e;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: 500;
    border: none;
    border-bottom: 2px solid transparent;
    margin-right: 8px;
}

QTabBar#topTabBar::tab:hover {
    color: #e8eaed;
    background: #2b2d31;
}

QTabBar#topTabBar::tab:selected {
    color: #4F8EF7;
    border-bottom: 2px solid #4F8EF7;
}

/* Planner Panel */
QFrame#plannerPanel {
    background-color: #1a1c1e;
}

QLineEdit#taskInput {
    background-color: #2b2d31;
    color: #e8eaed;
    border: none;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 14px;
}

QLineEdit#taskInput:focus {
    background-color: #33363b;
}

QPushButton#addTaskBtn {
    background-color: #4F8EF7;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 20px;
    font-size: 14px;
    font-weight: 500;
}

QPushButton#addTaskBtn:hover {
    background-color: #3d7be5;
}

QListWidget#taskList {
    background-color: transparent;
    border: none;
    outline: none;
    padding: 10px;
}

QListWidget#taskList::item {
    background-color: #2b2d31;
    color: #e8eaed;
    border-radius: 8px;
    padding: 0px;
    margin: 6px 10px;
    min-height: 48px;
}

QListWidget#taskList::item:hover {
    background-color: #33363b;
}

QListWidget#taskList::item:selected {
    background-color: #343541;
    border: 1px solid #4F8EF7;
}

QCheckBox#taskCheckbox {
    color: #e8eaed;
    font-size: 14px;
    spacing: 10px;
}

QCheckBox#taskCheckbox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    background-color: #3d3d3d;
    border: 1px solid #5d5d5d;
}

QCheckBox#taskCheckbox::indicator:checked {
    background-color: #4F8EF7;
    border: 1px solid #4F8EF7;
}

QCheckBox#taskCheckbox::indicator:hover {
    border: 1px solid #4F8EF7;
}
"""
