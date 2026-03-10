"""Modern dark theme — refined, minimal, state-of-the-art."""

COLORS = {
    # Backgrounds — rich dark with subtle blue undertone
    "bg":              "#0f1117",
    "bg_alt":          "#161822",
    "surface":         "#1c1e2d",
    "surface_hover":   "#252839",
    "surface_border":  "#2d3045",

    # Text — high contrast hierarchy
    "text":            "#f0f1f5",
    "text_secondary":  "#8b8fa3",
    "text_muted":      "#555970",

    # Accent — vibrant indigo-violet
    "accent":          "#6366f1",
    "accent_hover":    "#818cf8",
    "accent_subtle":   "#1e1b4b",
    "accent_glow":     "#4f46e520",

    # Semantic
    "success":         "#34d399",
    "success_bg":      "#0d2e22",
    "warning":         "#fbbf24",
    "warning_bg":      "#2e2509",
    "danger":          "#f87171",
    "danger_bg":       "#2e0f0f",

    # Borders & inputs
    "border":          "#262938",
    "border_subtle":   "#1f2130",
    "input_bg":        "#12141e",

    # Scrollbar
    "scrollbar":       "#2d3045",
    "scrollbar_hover": "#3d4157",

    # Tabs
    "tab_active":      "#6366f1",
    "tab_inactive":    "#161822",

    # Headers
    "header":          "#e0e2f0",
}


def get_stylesheet() -> str:
    c = COLORS
    return f"""
    QMainWindow, QWidget {{
        background-color: {c['bg']}; color: {c['text']};
        font-family: 'Inter', 'SF Pro Display', 'Segoe UI', 'Helvetica Neue', sans-serif;
        font-size: 13px;
    }}
    QLabel {{ color: {c['text']}; background: transparent; }}

    /* --- Inputs --- */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {c['input_bg']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 8px;
        padding: 7px 12px; min-height: 22px;
        selection-background-color: {c['accent']};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {c['accent']};
    }}
    QLineEdit::placeholder {{ color: {c['text_muted']}; }}
    QComboBox::drop-down {{ border: none; padding-right: 10px; }}
    QComboBox QAbstractItemView {{
        background-color: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; selection-background-color: {c['accent']};
        padding: 4px; border-radius: 8px;
    }}
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background-color: {c['surface']}; border: none; width: 20px;
    }}
    QSpinBox::up-button:hover, QSpinBox::down-button:hover,
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {c['surface_hover']};
    }}

    /* --- Buttons --- */
    QPushButton {{
        background-color: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 8px;
        padding: 8px 18px; font-weight: 500; min-height: 22px;
    }}
    QPushButton:hover {{
        background-color: {c['surface_hover']}; border-color: {c['accent']};
    }}
    QPushButton:pressed {{ background-color: {c['accent_subtle']}; }}
    QPushButton:disabled {{
        color: {c['text_muted']}; background-color: {c['bg_alt']};
        border-color: {c['border_subtle']};
    }}

    /* --- Tables --- */
    QTableWidget, QTableView {{
        background-color: {c['bg_alt']}; alternate-background-color: {c['surface']};
        color: {c['text']}; border: 1px solid {c['border']}; border-radius: 10px;
        gridline-color: transparent;
        selection-background-color: {c['accent_subtle']};
        selection-color: {c['text']};
    }}
    QTableWidget::item, QTableView::item {{
        padding: 5px 10px; border-bottom: 1px solid {c['border_subtle']};
    }}
    QTableWidget::item:selected, QTableView::item:selected {{
        background-color: {c['accent_subtle']};
    }}
    QHeaderView::section {{
        background-color: {c['bg_alt']}; color: {c['text_secondary']};
        padding: 8px 10px; border: none;
        border-bottom: 1px solid {c['border']};
        font-weight: 600; font-size: 11px;
        text-transform: uppercase; letter-spacing: 0.5px;
    }}

    /* --- Tabs --- */
    QTabWidget::pane {{
        border: 1px solid {c['border']}; border-radius: 10px;
        background-color: {c['bg_alt']}; top: -1px;
    }}
    QTabBar::tab {{
        background-color: transparent; color: {c['text_muted']};
        border: none; border-bottom: 2px solid transparent;
        padding: 10px 22px; margin-right: 4px; font-weight: 500;
    }}
    QTabBar::tab:selected {{
        color: {c['text']}; border-bottom: 2px solid {c['accent']};
    }}
    QTabBar::tab:hover:!selected {{
        color: {c['text_secondary']};
    }}

    /* --- Scrollbar --- */
    QScrollBar:vertical {{
        background-color: transparent; width: 8px; margin: 4px 2px;
    }}
    QScrollBar::handle:vertical {{
        background-color: {c['scrollbar']}; border-radius: 4px; min-height: 32px;
    }}
    QScrollBar::handle:vertical:hover {{ background-color: {c['scrollbar_hover']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{
        background-color: transparent; height: 8px; margin: 2px 4px;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {c['scrollbar']}; border-radius: 4px; min-width: 32px;
    }}
    QScrollBar::handle:horizontal:hover {{ background-color: {c['scrollbar_hover']}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    /* --- Progress --- */
    QProgressBar {{
        background-color: {c['surface']}; border: none;
        border-radius: 6px; text-align: center; color: {c['text']};
        min-height: 14px; max-height: 14px; font-size: 10px;
    }}
    QProgressBar::chunk {{
        background-color: {c['accent']}; border-radius: 6px;
    }}

    /* --- Text Edit --- */
    QTextEdit {{
        background-color: {c['input_bg']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 10px;
        padding: 10px; selection-background-color: {c['accent']};
    }}

    /* --- Splitter --- */
    QSplitter::handle {{
        background-color: {c['border_subtle']}; width: 1px;
    }}
    QSplitter::handle:hover {{ background-color: {c['accent']}; }}

    /* --- Status Bar --- */
    QStatusBar {{
        background-color: {c['bg']}; color: {c['text_secondary']};
        border-top: 1px solid {c['border_subtle']}; padding: 6px 12px;
        font-size: 12px;
    }}

    /* --- Misc --- */
    QScrollArea {{ background-color: transparent; border: none; }}
    QDialog {{ background-color: {c['bg']}; }}
    QMessageBox {{ background-color: {c['bg']}; }}
    QMessageBox QLabel {{ color: {c['text']}; }}

    QCheckBox {{ color: {c['text']}; background: transparent; spacing: 8px; }}
    QCheckBox::indicator {{
        width: 18px; height: 18px; border: 2px solid {c['border']};
        border-radius: 4px; background-color: {c['input_bg']};
    }}
    QCheckBox::indicator:hover {{ border-color: {c['accent']}; }}
    QCheckBox::indicator:checked {{
        background-color: {c['accent']}; border-color: {c['accent']};
    }}

    /* --- Group Box (training tab) --- */
    QGroupBox {{
        background-color: {c['bg_alt']}; border: 1px solid {c['border']};
        border-radius: 10px; margin-top: 14px; padding: 16px 12px 12px 12px;
        font-weight: 600; color: {c['text_secondary']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; subcontrol-position: top left;
        padding: 2px 12px; color: {c['text_secondary']};
        font-size: 12px; font-weight: 600;
    }}
    """


# --- Reusable style constants ---

CARD_STYLE = (
    f"background-color: {COLORS['bg_alt']}; "
    f"border: 1px solid {COLORS['border']}; "
    f"border-radius: 12px; padding: 14px;"
)

SECTION_HEADER_STYLE = (
    f"color: {COLORS['header']}; font-size: 14px; font-weight: 700; "
    f"padding: 6px 0 2px 0; background: transparent; letter-spacing: 0.3px;"
)

SECTION_SUBHEADER_STYLE = (
    f"color: {COLORS['text_secondary']}; font-size: 12px; font-weight: 600; "
    f"padding: 2px 0; background: transparent; text-transform: uppercase; "
    f"letter-spacing: 0.8px;"
)

ACCENT_BUTTON_STYLE = (
    f"QPushButton {{ background-color: {COLORS['accent']}; color: white; "
    f"border: none; border-radius: 8px; padding: 9px 22px; font-weight: 600; }} "
    f"QPushButton:hover {{ background-color: {COLORS['accent_hover']}; }} "
    f"QPushButton:pressed {{ background-color: {COLORS['accent_subtle']}; }}"
)

SUCCESS_BUTTON_STYLE = (
    f"QPushButton {{ background-color: {COLORS['success']}; color: {COLORS['bg']}; "
    f"border: none; border-radius: 8px; padding: 9px 22px; font-weight: 700; }} "
    f"QPushButton:hover {{ background-color: #5ce0a8; }}"
)

DANGER_BUTTON_STYLE = (
    f"QPushButton {{ background-color: transparent; color: {COLORS['danger']}; "
    f"border: 1px solid {COLORS['danger']}; border-radius: 8px; "
    f"padding: 8px 18px; font-weight: 500; }} "
    f"QPushButton:hover {{ background-color: {COLORS['danger_bg']}; }}"
)

SECURITY_BANNER_STYLE = (
    f"background-color: {COLORS['success_bg']}; color: {COLORS['success']}; "
    f"border: 1px solid #1a4a35; border-radius: 10px; "
    f"padding: 10px 18px; font-weight: 500; font-size: 12px;"
)

MUTED_LABEL_STYLE = (
    f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
)

STAT_VALUE_STYLE = (
    f"color: {COLORS['accent']}; font-size: 22px; font-weight: 700; "
    f"background: transparent;"
)

STAT_LABEL_STYLE = (
    f"color: {COLORS['text_muted']}; font-size: 10px; font-weight: 600; "
    f"background: transparent; text-transform: uppercase; letter-spacing: 0.5px;"
)

# Badge styles
TAG_BADGE_STYLE = (
    f"background-color: {COLORS['accent_subtle']}; color: {COLORS['accent_hover']}; "
    f"border-radius: 10px; padding: 3px 12px; "
    f"font-size: 11px; font-weight: 600;"
)

# Pill navigation button
NAV_BUTTON_STYLE = (
    f"QPushButton {{ background-color: {COLORS['surface']}; color: {COLORS['text']}; "
    f"border: 1px solid {COLORS['border']}; border-radius: 8px; "
    f"padding: 8px 20px; font-weight: 500; }} "
    f"QPushButton:hover {{ background-color: {COLORS['surface_hover']}; "
    f"border-color: {COLORS['accent']}; }} "
    f"QPushButton:disabled {{ color: {COLORS['text_muted']}; "
    f"background-color: {COLORS['bg']}; border-color: {COLORS['border_subtle']}; }}"
)
