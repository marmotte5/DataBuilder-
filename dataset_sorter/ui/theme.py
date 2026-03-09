"""Modern dark theme for the application."""

COLORS = {
    "bg":              "#1a1b2e",
    "bg_alt":          "#232440",
    "surface":         "#2a2b4a",
    "surface_hover":   "#33345a",
    "surface_border":  "#3d3e6b",
    "text":            "#e8e9f0",
    "text_secondary":  "#9a9bc0",
    "text_muted":      "#6b6c8a",
    "accent":          "#7c5cfc",
    "accent_hover":    "#9478ff",
    "accent_subtle":   "#3a2d6e",
    "success":         "#3dd68c",
    "success_bg":      "#1a3d2e",
    "warning":         "#f5a623",
    "warning_bg":      "#3d3020",
    "danger":          "#f55a5a",
    "danger_bg":       "#3d1a1a",
    "border":          "#3d3e6b",
    "input_bg":        "#1e1f36",
    "scrollbar":       "#4a4b6e",
    "scrollbar_hover": "#5a5b7e",
    "tab_active":      "#7c5cfc",
    "tab_inactive":    "#2a2b4a",
    "header":          "#e0d0ff",
}


def get_stylesheet() -> str:
    c = COLORS
    return f"""
    QMainWindow, QWidget {{
        background-color: {c['bg']}; color: {c['text']};
        font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', sans-serif;
        font-size: 13px;
    }}
    QLabel {{ color: {c['text']}; background: transparent; }}
    QLineEdit, QSpinBox, QComboBox {{
        background-color: {c['input_bg']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 6px;
        padding: 6px 10px; min-height: 20px;
        selection-background-color: {c['accent']};
    }}
    QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border-color: {c['accent']}; }}
    QLineEdit::placeholder {{ color: {c['text_muted']}; }}
    QComboBox::drop-down {{ border: none; padding-right: 8px; }}
    QComboBox QAbstractItemView {{
        background-color: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; selection-background-color: {c['accent']};
        padding: 4px;
    }}
    QSpinBox::up-button, QSpinBox::down-button {{
        background-color: {c['surface']}; border: none; width: 18px;
    }}
    QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
        background-color: {c['surface_hover']};
    }}
    QPushButton {{
        background-color: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 6px;
        padding: 7px 16px; font-weight: 500; min-height: 20px;
    }}
    QPushButton:hover {{ background-color: {c['surface_hover']}; border-color: {c['accent']}; }}
    QPushButton:pressed {{ background-color: {c['accent_subtle']}; }}
    QPushButton:disabled {{ color: {c['text_muted']}; background-color: {c['bg_alt']}; border-color: {c['bg_alt']}; }}
    QTableWidget, QTableView {{
        background-color: {c['bg_alt']}; alternate-background-color: {c['surface']};
        color: {c['text']}; border: 1px solid {c['border']}; border-radius: 8px;
        gridline-color: {c['border']}; selection-background-color: {c['accent_subtle']};
        selection-color: {c['text']};
    }}
    QTableWidget::item, QTableView::item {{ padding: 4px 8px; }}
    QTableWidget::item:selected, QTableView::item:selected {{ background-color: {c['accent_subtle']}; }}
    QHeaderView::section {{
        background-color: {c['surface']}; color: {c['header']};
        padding: 6px 8px; border: none; border-bottom: 2px solid {c['accent']};
        font-weight: 600;
    }}
    QTabWidget::pane {{
        border: 1px solid {c['border']}; border-radius: 8px;
        background-color: {c['bg_alt']}; top: -1px;
    }}
    QTabBar::tab {{
        background-color: {c['tab_inactive']}; color: {c['text_secondary']};
        border: 1px solid {c['border']}; border-bottom: none;
        border-top-left-radius: 8px; border-top-right-radius: 8px;
        padding: 8px 20px; margin-right: 2px; font-weight: 500;
    }}
    QTabBar::tab:selected {{ background-color: {c['bg_alt']}; color: {c['text']}; border-bottom: 2px solid {c['accent']}; }}
    QTabBar::tab:hover:!selected {{ background-color: {c['surface_hover']}; }}
    QScrollBar:vertical {{ background-color: transparent; width: 10px; }}
    QScrollBar::handle:vertical {{ background-color: {c['scrollbar']}; border-radius: 5px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background-color: {c['scrollbar_hover']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{ background-color: transparent; height: 10px; }}
    QScrollBar::handle:horizontal {{ background-color: {c['scrollbar']}; border-radius: 5px; min-width: 30px; }}
    QScrollBar::handle:horizontal:hover {{ background-color: {c['scrollbar_hover']}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QProgressBar {{
        background-color: {c['bg_alt']}; border: 1px solid {c['border']};
        border-radius: 6px; text-align: center; color: {c['text']}; min-height: 18px;
    }}
    QProgressBar::chunk {{ background-color: {c['accent']}; border-radius: 5px; }}
    QTextEdit {{
        background-color: {c['input_bg']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 6px;
        padding: 6px; selection-background-color: {c['accent']};
    }}
    QSplitter::handle {{ background-color: {c['border']}; width: 2px; }}
    QSplitter::handle:hover {{ background-color: {c['accent']}; }}
    QStatusBar {{
        background-color: {c['bg_alt']}; color: {c['text_secondary']};
        border-top: 1px solid {c['border']}; padding: 4px 8px;
    }}
    QScrollArea {{ background-color: transparent; border: none; }}
    QDialog {{ background-color: {c['bg']}; }}
    QMessageBox {{ background-color: {c['bg']}; }}
    QMessageBox QLabel {{ color: {c['text']}; }}
    QCheckBox {{ color: {c['text']}; background: transparent; spacing: 6px; }}
    QCheckBox::indicator {{
        width: 16px; height: 16px; border: 1px solid {c['border']};
        border-radius: 3px; background-color: {c['input_bg']};
    }}
    QCheckBox::indicator:checked {{ background-color: {c['accent']}; border-color: {c['accent']}; }}
    """


CARD_STYLE = f"background-color: {COLORS['bg_alt']}; border: 1px solid {COLORS['border']}; border-radius: 10px; padding: 12px;"
SECTION_HEADER_STYLE = f"color: {COLORS['header']}; font-size: 15px; font-weight: 700; padding: 4px 0; background: transparent;"
SECTION_SUBHEADER_STYLE = f"color: {COLORS['accent']}; font-size: 13px; font-weight: 600; padding: 2px 0; background: transparent;"
ACCENT_BUTTON_STYLE = f"QPushButton {{ background-color: {COLORS['accent']}; color: white; border: none; border-radius: 6px; padding: 8px 20px; font-weight: 600; }} QPushButton:hover {{ background-color: {COLORS['accent_hover']}; }} QPushButton:pressed {{ background-color: {COLORS['accent_subtle']}; }}"
SUCCESS_BUTTON_STYLE = f"QPushButton {{ background-color: {COLORS['success']}; color: {COLORS['bg']}; border: none; border-radius: 6px; padding: 8px 20px; font-weight: 700; }} QPushButton:hover {{ background-color: #4ae09a; }}"
DANGER_BUTTON_STYLE = f"QPushButton {{ background-color: transparent; color: {COLORS['danger']}; border: 1px solid {COLORS['danger']}; border-radius: 6px; padding: 7px 16px; font-weight: 500; }} QPushButton:hover {{ background-color: {COLORS['danger_bg']}; }}"
SECURITY_BANNER_STYLE = f"background-color: {COLORS['success_bg']}; color: {COLORS['success']}; border: 1px solid {COLORS['success']}; border-radius: 8px; padding: 10px 16px; font-weight: 600; font-size: 12px;"
MUTED_LABEL_STYLE = f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
STAT_VALUE_STYLE = f"color: {COLORS['accent']}; font-size: 20px; font-weight: 700; background: transparent;"
STAT_LABEL_STYLE = f"color: {COLORS['text_secondary']}; font-size: 11px; font-weight: 500; background: transparent;"
