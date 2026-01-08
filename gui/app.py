"""
Main PySide6 application setup and layout.
"""

import threading
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QFrame,
    QStackedWidget, QTabBar
)
from PySide6.QtCore import Qt, QTimer

from core.llm import preload_models
from core.tts import tts
from gui.handlers import ChatHandlers
from gui.styles import STYLESHEET
from gui.tabs.chat import ChatTab
from gui.tabs.planner import PlannerTab
from gui.tabs.briefing import BriefingView


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("A.D.A")
        self.setMinimumSize(1000, 700)
        self.resize(1000, 700)
        
        # Apply global stylesheet
        self.setStyleSheet(STYLESHEET)
        
        # Initialize handlers
        # Handlers are initialized before UI because they store state, 
        # but they rely on UI methods which we proxy below.
        self.handlers = ChatHandlers(self)
        
        self._setup_ui()
        self._connect_signals()
        self._init_background()
        
    def _setup_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # --- Top Tab Bar (persistent across all views) ---
        tab_bar_container = QFrame()
        tab_bar_container.setObjectName("chatPanel")
        tab_bar_layout = QVBoxLayout(tab_bar_container)
        tab_bar_layout.setContentsMargins(0, 0, 0, 0)
        tab_bar_layout.setSpacing(0)
        
        self.top_tab_bar = QTabBar()
        self.top_tab_bar.setObjectName("topTabBar")
        self.top_tab_bar.addTab("💬  Chat")
        self.top_tab_bar.addTab("📋  Planner")
        self.top_tab_bar.addTab("📰  Briefing")
        self.top_tab_bar.setExpanding(False)
        tab_bar_layout.addWidget(self.top_tab_bar)
        
        # Tab bar divider
        tab_divider = QFrame()
        tab_divider.setFixedHeight(1)
        tab_divider.setStyleSheet("background-color: #3d3d3d;")
        tab_bar_layout.addWidget(tab_divider)
        
        main_layout.addWidget(tab_bar_container)
        
        # --- Stacked Widget for tab content ---
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack)
        
        # --- Page 0: Chat ---
        self.chat_tab = ChatTab()
        self.content_stack.addWidget(self.chat_tab)
        
        # --- Page 1: Planner ---
        self.planner_tab = PlannerTab()
        self.content_stack.addWidget(self.planner_tab)
        
        # --- Page 2: Briefing ---
        self.briefing_view = BriefingView()
        self.content_stack.addWidget(self.briefing_view)
    
    def _connect_signals(self):
        """Connect signals between UI components and logic."""
        self.top_tab_bar.currentChanged.connect(self._on_tab_changed)
        
        # Chat Logic Connections
        self.chat_tab.new_chat_requested.connect(self.handlers.clear_chat)
        self.chat_tab.send_message_requested.connect(self._on_send)
        self.chat_tab.stop_generation_requested.connect(self.handlers.stop_generation)
        self.chat_tab.tts_toggled.connect(self.handlers.toggle_tts)
        self.chat_tab.session_selected.connect(self._on_session_clicked)
        
        # Session Management
        self.chat_tab.session_pin_requested.connect(self.handlers.pin_session)
        self.chat_tab.session_rename_requested.connect(self.handlers.rename_session)
        self.chat_tab.session_delete_requested.connect(self.handlers.delete_session)

    def _on_tab_changed(self, index: int):
        """Handle tab bar selection changes."""
        self.content_stack.setCurrentIndex(index)

    def _on_send(self, text):
        """Forward send request to handlers."""
        self.handlers.send_message(text)
        
    def _on_session_clicked(self, session_id):
        """Load session."""
        self.handlers.load_session(session_id)
    
    def _init_background(self):
        """Initialize models in background."""
        def preload_background():
            self.set_status("Warming up models...")
            preload_models()
            if tts.toggle(True):
                self.set_status("Ready | TTS Active")
            else:
                self.set_status("Ready | TTS Failed")
        
        threading.Thread(target=preload_background, daemon=True).start()
        # Refresh sidebar specifically on the chat tab
        self.chat_tab.refresh_sidebar()
    
    # --- Public Methods for Handlers (Facade Pattern) ---
    # These methods proxy calls to the unified ChatTab component
    # allowing ChatHandlers to remain unchanged for now.
    
    def set_status(self, text: str):
        self.chat_tab.set_status(text)
    
    def clear_input(self):
        self.chat_tab.clear_input()
    
    def set_generating_state(self, is_generating: bool):
        self.chat_tab.set_generating_state(is_generating)
    
    def add_message_bubble(self, role: str, text: str, is_thinking: bool = False):
        self.chat_tab.add_message_bubble(role, text, is_thinking)
    
    def add_streaming_widgets(self, thinking_ui, response_bubble):
        self.chat_tab.add_streaming_widgets(thinking_ui, response_bubble)
    
    def clear_chat_display(self):
        self.chat_tab.clear_chat_display()
    
    def refresh_sidebar(self, current_session_id: str = None):
        self.chat_tab.refresh_sidebar(current_session_id)
    
    def scroll_to_bottom(self):
        self.chat_tab.scroll_to_bottom()


def create_app():
    """Create and return the main window."""
    return MainWindow()

