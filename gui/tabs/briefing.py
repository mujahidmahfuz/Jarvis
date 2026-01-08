from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QFrame, QPushButton, QGridLayout
)
from PySide6.QtCore import Qt, QThread, Signal
from gui.components.news_card import NewsCard
from core.news import news_manager

class NewsLoaderThread(QThread):
    loaded = Signal(list)
    status_update = Signal(str)
    
    def run(self):
        # Pass a lambda that emits the signal back to the main thread
        news = news_manager.get_briefing(status_callback=self.status_update.emit)
        self.loaded.emit(news)

class BriefingView(QWidget):
    """
    The main Dashboard/Briefing view.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("briefingView")
        self.setStyleSheet("background-color: transparent;")
        
        # Main Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(30, 30, 30, 30)
        self.layout.setSpacing(20)
        
        # Header Area
        header_layout = QHBoxLayout()
        
        title_block = QVBoxLayout()
        title = QLabel("Briefing")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #e8eaed; letter-spacing: 1px;")
        
        subtitle = QLabel("Curated intelligence from global sources.")
        subtitle.setStyleSheet("font-size: 14px; color: #9e9e9e;")
        
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header_layout.addLayout(title_block)
        
        header_layout.addStretch()
        
        # Refresh Button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(79, 142, 247, 0.2);
                color: #4F8EF7;
                border: 1px solid #4F8EF7;
                padding: 8px 16px;
                border-radius: 8px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: rgba(79, 142, 247, 0.3);
            }
        """)
        refresh_btn.clicked.connect(self.load_news)
        header_layout.addWidget(refresh_btn)
        
        self.layout.addLayout(header_layout)
        
        # Breaking News Ticker (Placeholder)
        self.breaking_widget = QFrame()
        self.breaking_widget.setStyleSheet("""
            background-color: rgba(239, 83, 80, 0.1);
            border: 1px solid rgba(239, 83, 80, 0.3);
            border-radius: 8px;
        """)
        self.breaking_widget.setFixedHeight(50)
        bk_layout = QHBoxLayout(self.breaking_widget)
        bk_layout.setContentsMargins(15, 0, 15, 0)
        
        bk_label = QLabel("BREAKING")
        bk_label.setStyleSheet("background-color: #ef5350; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;")
        bk_layout.addWidget(bk_label)
        
        self.bk_text = QLabel("Loading latest intelligence stream...")
        self.bk_text.setStyleSheet("color: #ffa4a2; font-size: 13px; font-weight: 500;")
        bk_layout.addWidget(self.bk_text)
        bk_layout.addStretch()
        
        self.layout.addWidget(self.breaking_widget)
        
        # Category Filters (Tabs)
        self.tabs_layout = QHBoxLayout()
        self.tabs_layout.setSpacing(10)
        
        categories = ["Top Stories", "Technology", "Markets", "Science", "Culture"]
        self.cats = []
        for c in categories:
            btn = QPushButton(c)
            btn.setCheckable(True)
            if c == "Top Stories": btn.setChecked(True)
            btn.setCursor(Qt.PointingHandCursor)
            # Styling would need to handle 'checked' state
            btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #9e9e9e;
                    border: none;
                    font-size: 14px;
                    font-weight: 500;
                    padding: 8px 12px;
                }
                QPushButton:hover { color: #e8eaed; }
                QPushButton:checked {
                    background-color: rgba(255,255,255,0.1);
                    color: white;
                    border-radius: 16px;
                }
            """)
            self.tabs_layout.addWidget(btn)
            self.cats.append(btn)
            
        self.tabs_layout.addStretch()
        self.layout.addLayout(self.tabs_layout)
        
        # News Grid (Scroll Area)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        
        container = QWidget()
        self.grid_layout = QGridLayout(container)
        self.grid_layout.setSpacing(20)
        self.grid_layout.setContentsMargins(0, 0, 0, 20)
        self.grid_layout.setAlignment(Qt.AlignTop)
        
        scroll.setWidget(container)
        self.layout.addWidget(scroll)
        
        # Load Data
        self.load_news()

    def load_news(self):
        self.bk_text.setText("Syncing global sources...")
        
        # Clear grid safely
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            
        # Start thread
        self.thread = NewsLoaderThread()
        self.thread.status_update.connect(self.bk_text.setText)
        self.thread.loaded.connect(self.display_news)
        self.thread.start()
        
    def display_news(self, news_items):
        if not news_items:
            self.bk_text.setText("System offline. No news available.")
            return
            
        # Update breaking news with the first item
        if news_items:
            first = news_items[0]
            self.bk_text.setText(f"{first['title']} ({first['source']})")
        
        # Populate Grid (2 columns)
        row = 0
        col = 0
        for item in news_items:
            card = NewsCard(item)
            self.grid_layout.addWidget(card, row, col)
            
            col += 1
            if col > 1:
                col = 0
                row += 1
