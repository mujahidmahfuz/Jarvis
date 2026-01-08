from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, 
    QPushButton, QListWidget, QListWidgetItem, QScrollArea, 
    QLineEdit, QSizePolicy, QMenu, QInputDialog
)
from PySide6.QtCore import Qt, QSize, QTimer, Signal
from PySide6.QtGui import QFont, QIcon

from gui.components import MessageBubble, ThinkingExpander, ToggleSwitch
from core.history import history_manager


class ChatTab(QWidget):
    """
    Chat Tab Component.
    Contains the Sidebar (Sessions) and Chat Area.
    """
    
    # Signals to communicate with MainWindow/Handlers
    send_message_requested = Signal(str)
    stop_generation_requested = Signal()
    tts_toggled = Signal(bool)
    new_chat_requested = Signal()
    session_selected = Signal(str)
    
    # Session handling signals (proxy to handlers)
    session_pin_requested = Signal(str)
    session_rename_requested = Signal(str, str)
    session_delete_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("ChatTab")
        self._setup_ui()
        self._connect_internal_signals()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Sidebar ---
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(300)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Title
        title_label = QLabel("A.D.A")
        title_label.setObjectName("sidebarTitle")
        sidebar_layout.addWidget(title_label)

        # Divider
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #3d3d3d;")
        sidebar_layout.addWidget(divider)

        # New Chat Button
        new_chat_container = QWidget()
        new_chat_layout = QHBoxLayout(new_chat_container)
        new_chat_layout.setContentsMargins(10, 10, 10, 5)

        self.new_chat_btn = QPushButton("➕  New Chat")
        self.new_chat_btn.setObjectName("newChatBtn")
        new_chat_layout.addWidget(self.new_chat_btn)
        sidebar_layout.addWidget(new_chat_container)

        # Session List
        self.session_list = QListWidget()
        self.session_list.setObjectName("sessionList")
        self.session_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.session_list.customContextMenuRequested.connect(self._show_session_context_menu)
        sidebar_layout.addWidget(self.session_list)

        layout.addWidget(self.sidebar)

        # --- Chat Content Area ---
        chat_content = QFrame()
        chat_content.setObjectName("chatPanel")
        chat_layout = QVBoxLayout(chat_content)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setObjectName("chatHeader")
        header.setFixedHeight(50)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)

        header_title = QLabel("Chat")
        header_title.setObjectName("headerTitle")
        header_layout.addWidget(header_title)

        header_layout.addStretch()

        # TTS Toggle
        tts_label = QLabel("Voice")
        tts_label.setStyleSheet("color: #9e9e9e; font-size: 12px;")
        header_layout.addWidget(tts_label)

        self.tts_toggle = ToggleSwitch(checked=False)
        header_layout.addWidget(self.tts_toggle)

        chat_layout.addWidget(header)

        # Status
        self.status_label = QLabel("Initializing...")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setContentsMargins(20, 5, 20, 5)
        chat_layout.addWidget(self.status_label)

        # Divider
        divider2 = QFrame()
        divider2.setFixedHeight(1)
        divider2.setStyleSheet("background-color: #3d3d3d;")
        chat_layout.addWidget(divider2)

        # Chat Scroll Area
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setObjectName("chatScroll")
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.chat_container = QWidget()
        self.chat_container_layout = QVBoxLayout(self.chat_container) # Renamed to avoid confusion
        self.chat_container_layout.setContentsMargins(20, 20, 20, 20)
        self.chat_container_layout.setSpacing(15)
        self.chat_container_layout.addStretch()

        self.chat_scroll.setWidget(self.chat_container)
        chat_layout.addWidget(self.chat_scroll)

        # Input Bar
        input_bar = QFrame()
        input_bar.setObjectName("inputBar")
        input_bar.setFixedHeight(76)
        input_layout = QHBoxLayout(input_bar)
        input_layout.setContentsMargins(15, 16, 15, 16)
        input_layout.setSpacing(12)
        input_layout.setAlignment(Qt.AlignVCenter)

        self.user_input = QLineEdit()
        self.user_input.setObjectName("userInput")
        self.user_input.setPlaceholderText("Ask something...")
        input_layout.addWidget(self.user_input)

        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setVisible(False)
        self.stop_btn.setToolTip("Stop Generation")
        input_layout.addWidget(self.stop_btn)

        self.send_btn = QPushButton("➤")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.setToolTip("Send")
        input_layout.addWidget(self.send_btn)

        chat_layout.addWidget(input_bar)

        layout.addWidget(chat_content)

    def _connect_internal_signals(self):
        """Connect internal UI events to public signals."""
        self.new_chat_btn.clicked.connect(self.new_chat_requested.emit)
        self.send_btn.clicked.connect(self._on_send_clicked)
        self.user_input.returnPressed.connect(self._on_send_clicked)
        self.stop_btn.clicked.connect(self.stop_generation_requested.emit)
        self.tts_toggle.toggled.connect(self.tts_toggled.emit)
        self.session_list.itemClicked.connect(self._on_session_clicked)

    def _on_send_clicked(self):
        text = self.user_input.text()
        self.send_message_requested.emit(text)

    def _on_session_clicked(self, item: QListWidgetItem):
        session_id = item.data(Qt.UserRole)
        if session_id:
            self.session_selected.emit(session_id)

    # --- Public API for Controller/MainWindow ---

    def set_status(self, text: str):
        """Update status label."""
        # Use QTimer to ensure thread safety if called from background thread
        QTimer.singleShot(0, lambda: self.status_label.setText(text))

    def clear_input(self):
        self.user_input.clear()

    def set_generating_state(self, is_generating: bool):
         """Switch states."""
         self.send_btn.setVisible(not is_generating)
         self.stop_btn.setVisible(is_generating)
         self.user_input.setEnabled(not is_generating)
         if not is_generating:
             self.user_input.setFocus()

    def add_message_bubble(self, role: str, text: str, is_thinking: bool = False):
        """Add a bubble."""
        bubble = MessageBubble(role, text, is_thinking)
        
        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        
        if role == "user":
            wrapper_layout.addStretch()
            wrapper_layout.addWidget(bubble)
        else:
            wrapper_layout.addWidget(bubble)
            wrapper_layout.addStretch()
        
        # Insert before stretch (last item)
        count = self.chat_container_layout.count()
        self.chat_container_layout.insertWidget(count - 1, wrapper)
        
        QTimer.singleShot(50, self.scroll_to_bottom)

    def add_streaming_widgets(self, thinking_ui, response_bubble):
        """Add streaming widgets."""
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(8)
        
        wrapper_layout.addWidget(thinking_ui)
        
        bubble_wrapper = QWidget()
        bubble_layout = QHBoxLayout(bubble_wrapper)
        bubble_layout.setContentsMargins(0, 0, 0, 0)
        bubble_layout.addWidget(response_bubble)
        bubble_layout.addStretch()
        wrapper_layout.addWidget(bubble_wrapper)
        
        count = self.chat_container_layout.count()
        self.chat_container_layout.insertWidget(count - 1, wrapper)
        
        QTimer.singleShot(50, self.scroll_to_bottom)

    def clear_chat_display(self):
        """Clear chat."""
        # Keep only the last stretch item
        while self.chat_container_layout.count() > 1:
            item = self.chat_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def scroll_to_bottom(self):
        scrollbar = self.chat_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def refresh_sidebar(self, current_session_id: str = None):
        """Refresh sidebar list."""
        self.session_list.clear()
        sessions = history_manager.get_sessions()
        
        for sess in sessions:
            title = sess['title']
            sid = sess['id']
            is_pinned = sess.get('pinned', False)
            is_current = sid == current_session_id
            
            # Custom widget
            item_widget = QWidget()
            item_widget.setFixedHeight(40)
            item_widget.setStyleSheet("background-color: transparent;")
            item_widget.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(8, 0, 8, 0)
            item_layout.setSpacing(8)
            
            icon = "📌" if is_pinned else "💬"
            icon_label = QLabel(icon)
            icon_label.setFixedWidth(20)
            icon_label.setStyleSheet("color: #9e9e9e;")
            item_layout.addWidget(icon_label)
            
            display_title = title[:35] + "..." if len(title) > 35 else title
            title_label = QLabel(display_title)
            title_label.setStyleSheet(f"color: {'white' if is_current else '#9e9e9e'}; font-size: 13px;")
            title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            item_layout.addWidget(title_label)
            
            # Menu Button
            menu_container = QWidget()
            menu_container.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            menu_layout = QHBoxLayout(menu_container)
            menu_layout.setContentsMargins(0, 0, 0, 0)
            
            menu_btn = QPushButton("⋮")
            menu_btn.setFixedSize(24, 24)
            menu_btn.setCursor(Qt.PointingHandCursor)
            menu_btn.setStyleSheet("QPushButton { background: transparent; color: #6e6e6e; font-size: 16px; border: none; } QPushButton:hover { color: #e8eaed; }")
            
            # Use closure to capture variables
            menu_btn.clicked.connect(lambda checked, s=sid, w=item_widget, p=is_pinned: self._show_item_menu(s, w, p))
            
            menu_layout.addWidget(menu_btn)
            item_layout.addWidget(menu_container)
            
            item = QListWidgetItem()
            item.setData(Qt.UserRole, sid)
            item.setSizeHint(QSize(280, 48))
            
            if is_current:
                item.setSelected(True)
            
            self.session_list.addItem(item)
            self.session_list.setItemWidget(item, item_widget)
            
        if not sessions:
            empty = QListWidgetItem("No conversations yet")
            empty.setFlags(Qt.NoItemFlags)
            self.session_list.addItem(empty)

    def _show_item_menu(self, session_id, widget, is_pinned):
        menu = QMenu(self)
        self._style_menu(menu)
        
        pin_text = "📌  Unpin" if is_pinned else "📌  Pin"
        menu.addAction(pin_text).triggered.connect(lambda: self.session_pin_requested.emit(session_id))
        menu.addAction("✏️  Rename").triggered.connect(lambda: self._prompt_rename(session_id))
        menu.addSeparator()
        menu.addAction("🗑️  Delete").triggered.connect(lambda: self.session_delete_requested.emit(session_id))
        
        menu.exec(widget.mapToGlobal(widget.rect().topRight()))

    def _show_session_context_menu(self, position):
        item = self.session_list.itemAt(position)
        if not item: return
        session_id = item.data(Qt.UserRole)
        if not session_id: return
        
        menu = QMenu(self)
        self._style_menu(menu)
        
        menu.addAction("📌  Pin/Unpin").triggered.connect(lambda: self.session_pin_requested.emit(session_id))
        menu.addAction("✏️  Rename").triggered.connect(lambda: self._prompt_rename(session_id))
        menu.addSeparator()
        menu.addAction("🗑️  Delete").triggered.connect(lambda: self.session_delete_requested.emit(session_id))
        
        menu.exec(self.session_list.mapToGlobal(position))

    def _prompt_rename(self, session_id):
        # We can handle the dialog here and just emit the result
        from PySide6.QtWidgets import QInputDialog
        new_title, ok = QInputDialog.getText(self, "Rename Chat", "Enter new name:")
        if ok and new_title.strip():
            self.session_rename_requested.emit(session_id, new_title.strip())

    def _style_menu(self, menu):
        menu.setStyleSheet("""
            QMenu { background-color: #2b2d31; border: 1px solid #3d3d3d; border-radius: 8px; padding: 5px; }
            QMenu::item { color: #e8eaed; padding: 8px 20px; border-radius: 4px; }
            QMenu::item:selected { background-color: #3d3d3d; }
            QMenu::separator { height: 1px; background: #3d3d3d; margin: 5px 10px; }
        """)
