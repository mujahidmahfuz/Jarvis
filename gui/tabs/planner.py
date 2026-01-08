from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, 
    QPushButton, QListWidget, QListWidgetItem, QCheckBox, 
    QInputDialog, QWidget, QSizePolicy
)
from PySide6.QtCore import Qt, QSize

from gui.components.schedule import ScheduleComponent
from gui.components.timer import TimerComponent
from gui.components.alarm import AlarmComponent
from core.tasks import task_manager


class PlannerTab(QFrame):
    """
    Planner functionality: Focus Tasks, Schedule, and Flow State tools.
    """
    
    def __init__(self):
        super().__init__()
        self.setObjectName("plannerPanel")
        self.setStyleSheet("""
            QFrame#plannerPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #121212, stop:1 #1e1e24);
            }
        """)
        
        self.completed_expanded = False
        
        self._setup_ui()
        self._load_tasks()

    def _setup_ui(self):
        planner_layout = QHBoxLayout(self)
        planner_layout.setContentsMargins(30, 30, 30, 30)
        planner_layout.setSpacing(25)
        
        # --- Column 1: Focus Tasks ---
        tasks_col = QFrame()
        tasks_col.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 35, 200);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
        """)
        tasks_layout = QVBoxLayout(tasks_col)
        tasks_layout.setContentsMargins(20, 25, 20, 25)
        tasks_layout.setSpacing(15)
        
        # Task Header
        t_title = QLabel("FOCUS TASKS")
        t_title.setStyleSheet("color: #e8eaed; font-size: 14px; font-weight: bold; letter-spacing: 1px; background: transparent; border: none;")
        tasks_layout.addWidget(t_title)
        
        # New Task Input (Top)
        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("Add a new task...")
        self.task_input.returnPressed.connect(self._add_task)
        self.task_input.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                color: white;
                padding: 10px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #bd93f9;
                background: rgba(189, 147, 249, 0.05);
            }
        """)
        tasks_layout.addWidget(self.task_input)
        
        # Divider
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet("background-color: rgba(255, 255, 255, 0.1); border: none;")
        tasks_layout.addWidget(div)
        
        # Task Lists (Active)
        self.task_list = QListWidget()
        self.task_list.setObjectName("taskList")
        self.task_list.setStyleSheet("background: transparent; border: none;") # Override global style
        tasks_layout.addWidget(self.task_list, 1) # Stretch to fill
        
        # Completed Section
        self.completed_header = QPushButton("▼  Completed  0")
        self.completed_header.setObjectName("completedHeader")
        self.completed_header.setCursor(Qt.PointingHandCursor)
        self.completed_header.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #6e6e6e;
                font-size: 12px;
                font-weight: 500;
                border: none;
                text-align: left;
                padding: 10px 0;
            }
            QPushButton:hover { color: #e8eaed; }
        """)
        self.completed_header.clicked.connect(self._toggle_completed_section)
        tasks_layout.addWidget(self.completed_header)
        
        self.completed_list = QListWidget()
        self.completed_list.setObjectName("taskList")
        self.completed_list.setStyleSheet("background: transparent; border: none;")
        self.completed_list.setVisible(False)
        tasks_layout.addWidget(self.completed_list)
        
        # Task Counter (Hidden/Small)
        self.task_counter = QLabel("0 tasks")
        self.task_counter.setStyleSheet("color: #6e6e6e; font-size: 11px; margin-top: 5px;")
        self.task_counter.setAlignment(Qt.AlignCenter)
        tasks_layout.addWidget(self.task_counter)
        
        planner_layout.addWidget(tasks_col, 1)
        
        # --- Column 2: Today's Schedule ---
        schedule_col = QFrame()
        schedule_col.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 35, 200);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
        """)
        schedule_col.setLayout(QVBoxLayout())
        schedule_col.layout().setContentsMargins(20, 25, 20, 25)
        
        self.schedule_component = ScheduleComponent()
        schedule_col.layout().addWidget(self.schedule_component)
        
        planner_layout.addWidget(schedule_col, 1)
        
        # --- Column 3: Flow State ---
        flow_col = QFrame()
        flow_col.setFixedWidth(320)
        flow_col.setStyleSheet("background: transparent; border: none;") # Wrapper is transparent
        flow_layout = QVBoxLayout(flow_col)
        flow_layout.setContentsMargins(0, 0, 0, 0)
        flow_layout.setSpacing(25)
        
        self.timer_component = TimerComponent()
        flow_layout.addWidget(self.timer_component)
        
        self.alarm_component = AlarmComponent()
        flow_layout.addWidget(self.alarm_component)
        
        planner_layout.addWidget(flow_col)

    def _load_tasks(self):
        """Load tasks from persistent storage."""
        tasks = task_manager.get_tasks()
        for task in tasks:
            self._create_task_item(task)
            
        self._update_task_counter()
        
    def _add_task(self):
        """Add a new task from the input field."""
        if hasattr(self, 'task_input'):
            task_text = self.task_input.text().strip()
            if task_text:
                self._add_task_from_text(task_text)
                self.task_input.clear()
    
    def _add_task_from_text(self, task_text):
        """Internal helper to add task."""
        # Save to DB
        new_task = task_manager.add_task(task_text)
        if new_task:
            self._create_task_item(new_task)
        self._update_task_counter()
    
    def _on_task_checked(self, state: int, item: QListWidgetItem, source_list: QListWidget):
        """Handle task checkbox state change - move between lists."""
        widget = source_list.itemWidget(item)
        if not widget:
            return
            
        # Get task ID from data
        task_id = item.data(Qt.UserRole)
        
        # Get task text from label
        label = widget.findChild(QLabel)
        if not label:
            return
        
        task_text = label.text()
        row = source_list.row(item)
        is_completed = (state == Qt.Checked.value)
        
        # Update persistence
        task_manager.toggle_task(task_id, is_completed)
        
        # Start transition
        source_list.takeItem(row)
        
        # Re-create in correct list
        task_data = {"id": task_id, "text": task_text, "completed": is_completed}
        self._create_task_item(task_data)
        
        self._update_task_counter()
    
    def _create_task_item(self, task_data: dict):
        """Create a task item widget and add to appropriate list."""
        completed = task_data.get('completed', False)
        text = task_data.get('text', '')
        task_id = task_data.get('id')
        
        target_list = self.completed_list if completed else self.task_list
        
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, 56))
        item.setData(Qt.UserRole, task_id)  # Store ID
        
        task_widget = QWidget()
        task_widget.setMinimumHeight(48)
        task_layout = QHBoxLayout(task_widget)
        task_layout.setContentsMargins(16, 12, 16, 12)
        task_layout.setSpacing(12)
        
        # Checkbox
        checkbox = QCheckBox()
        checkbox.setObjectName("taskCheckbox")
        checkbox.setChecked(completed)
        checkbox.stateChanged.connect(lambda state, i=item, l=target_list: self._on_task_checked(state, i, l))
        task_layout.addWidget(checkbox)
        
        # Task label
        task_label = QLabel(text)
        if completed:
            task_label.setStyleSheet("color: #6e6e6e; font-size: 14px; padding: 2px 0; text-decoration: line-through;")
        else:
            task_label.setStyleSheet("color: #e8eaed; font-size: 14px; padding: 2px 0;")
        task_label.setWordWrap(False)
        task_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        task_layout.addWidget(task_label, 1)
        
        # Delete button
        delete_btn = QPushButton("🗑️")
        delete_btn.setFixedSize(32, 32)
        delete_btn.setCursor(Qt.PointingHandCursor)
        delete_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #6e6e6e;
                font-size: 14px;
                border: none;
                border-radius: 16px;
            }
            QPushButton:hover {
                background: #3d3d3d;
                color: #ef5350;
            }
        """)
        delete_btn.clicked.connect(lambda: self._delete_task(item, target_list))
        task_layout.addWidget(delete_btn)
        
        target_list.addItem(item)
        target_list.setItemWidget(item, task_widget)
    
    def _delete_task(self, item: QListWidgetItem, source_list: QListWidget = None):
        """Delete a task from the list."""
        if source_list is None:
            source_list = self.task_list
            
        task_id = item.data(Qt.UserRole)
        task_manager.delete_task(task_id)
        
        row = source_list.row(item)
        if row >= 0:
            source_list.takeItem(row)
            self._update_task_counter()
    
    def _toggle_completed_section(self):
        """Toggle the completed tasks section visibility."""
        self.completed_expanded = not self.completed_expanded
        self.completed_list.setVisible(self.completed_expanded)
        
        # Update header arrow
        count = self.completed_list.count()
        arrow = "▼" if self.completed_expanded else "▶"
        self.completed_header.setText(f"{arrow}  Completed  {count}")
    
    def _update_task_counter(self):
        """Update the task counter label and completed header."""
        active_count = self.task_list.count()
        completed_count = self.completed_list.count()
        
        self.task_counter.setText(f"{active_count} active task{'s' if active_count != 1 else ''}")
        
        # Update completed header
        arrow = "▼" if self.completed_expanded else "▶"
        self.completed_header.setText(f"{arrow}  Completed  {completed_count}")
