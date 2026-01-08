
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from PySide6.QtWidgets import QApplication
try:
    from gui.app import MainWindow
except ImportError as e:
    with open("import_error.txt", "w") as f:
        f.write(str(e))
    print(f"ImportError: {e}")
    sys.exit(1)

def test_mainwindow_structure():
    print("Initializing QApplication...")
    # handle existing instance in case we are running in an env that already has one
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    print("Initializing MainWindow...")
    try:
        window = MainWindow()
        print("MainWindow created.")
        
        # Verify Tabs
        if not hasattr(window, 'briefing_view'):
            print("ERROR: briefing_view attribute missing")
            sys.exit(1)
        if not hasattr(window, 'planner_tab'):
            print("ERROR: planner_tab attribute missing")
            sys.exit(1)
            
        print("Verification Successful: All tabs initialized.")
    except Exception as e:
        error_msg = f"Runtime Error: {e}\n"
        import traceback
        error_msg += traceback.format_exc()
        print(error_msg)
        with open("test_log.txt", "w") as f:
            f.write(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    test_mainwindow_structure()
