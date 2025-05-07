import os
import sys
import platform
import traceback
import json
import logging
import psutil
import subprocess
import shutil
import winreg
from PySide6.QtWidgets import QMessageBox, QApplication

from config import LOG_FILE, LOG_FORMATTER, LOG_DATE_FORMAT

def setup_logging():
    log_formatter = logging.Formatter(LOG_FORMATTER, datefmt=LOG_DATE_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)
    
    logging.info("Démarrage de ChatGUI_AI_Local_API")

def check_vc_redist():
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64")
        value, _ = winreg.QueryValueEx(key, "Installed")
        return value == 1
    except Exception:
        return False

def check_python():
    return shutil.which("python") is not None

def check_dependencies():
    missing = []
    try:
        import PySide6
    except ImportError:
        missing.append("PySide6")
    try:
        import requests
    except ImportError:
        missing.append("requests")
    try:
        import psutil
    except ImportError:
        missing.append("psutil")
    try:
        import rich
    except ImportError:
        missing.append("rich")
    try:
        import dotenv
    except ImportError:
        missing.append("python-dotenv")
    try:
        import openai
    except ImportError:
        missing.append("openai")
    try:
        import httpx
    except ImportError:
        missing.append("httpx")
    return missing

def prompt_install(title, message, installer_path=None, url=None):
    app = QApplication.instance()
    btn = QMessageBox.question(None, title, message, QMessageBox.Yes | QMessageBox.No)
    if btn == QMessageBox.Yes:
        if installer_path and os.path.exists(installer_path):
            subprocess.Popen([installer_path], shell=True)
        elif url:
            import webbrowser
            webbrowser.open(url)

def log_critical_error(context: str, exc: Exception, extra: dict = None):
    sys_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    if extra:
        sys_info.update(extra)
    logging.error(
        f"{context}\n"
        f"Exception: {exc}\n"
        f"WinError: {getattr(exc, 'winerror', None)}, errno: {getattr(exc, 'errno', None)}\n"
        f"Traceback:\n{traceback.format_exc()}\n"
        f"System info: {json.dumps(sys_info, indent=2, ensure_ascii=False)}"
    )