"""Modern theme system — dark/light modes with runtime toggling.

Premium design language: deep layered surfaces, refined typography,
smooth transitions, and studio-grade visual polish.

The dark palette mirrors the DataBuilder splash artwork — a futuristic,
"data-cube" aesthetic with deep midnight surfaces, electric-cyan glows,
violet highlights, and neon-green status accents. Think
data-centre dashboard meets sci-fi HUD.
"""

_DARK_COLORS = {
    # Backgrounds — abyssal midnight with a faint blue undertone, the
    # exact tone of the splash's "data city" backdrop.
    "bg":              "#070b14",
    "bg_alt":          "#0c111d",
    "surface":         "#121829",
    "surface_hover":   "#1a2236",
    "surface_border":  "#1f2a44",

    # Text — clean cool-white hierarchy. Verified WCAG AA against `bg`.
    "text":            "#e6efff",
    "text_secondary":  "#8ea0c2",
    "text_muted":      "#7c8aac",

    # Accent — electric cyan, the unmistakable splash "neon" colour.
    "accent":          "#00d4ff",
    "accent_hover":    "#54e8ff",
    "accent_subtle":   "#0a2540",
    "accent_glow":     "#00d4ff26",

    # Semantic — vivid neon for status pills.
    "success":         "#22e3a4",
    "success_bg":      "#06241c",
    "warning":         "#ffc857",
    "warning_bg":      "#2a1f08",
    "danger":          "#ff5e7a",
    "danger_bg":       "#2c0d18",

    # Borders & inputs — barely-visible holographic edges.
    "border":          "#1d2742",
    "border_subtle":   "#141b30",
    "input_bg":        "#0a1020",

    # Scrollbar — thin and cyan-tinted on hover.
    "scrollbar":       "#1d2742",
    "scrollbar_hover": "#2c4373",

    # Tabs — active tab carries the cyan glow.
    "tab_active":      "#00d4ff",
    "tab_inactive":    "#0c111d",

    # Headers — pure white for the dashboard title vibe.
    "header":          "#f3f8ff",

    # Card elevation — layered like floating HUD panels.
    "card_bg":         "#0e1424",
    "card_hover":      "#15203a",
    "card_selected":   "#1a2a52",

    # Gradient accents — cyan → violet sweep, matching the splash's
    # "data beam" rays.
    "gradient_start":  "#00d4ff",
    "gradient_end":    "#a36bff",

    # Category badge colors — picked straight from the splash glyphs.
    "badge_model":     "#54e8ff",
    "badge_lora":      "#22e3a4",
    "badge_embedding": "#ffc857",
    "badge_text":      "#070b14",
}

_LIGHT_COLORS = {
    # Backgrounds — warm pearl white
    "bg":              "#f5f6fa",
    "bg_alt":          "#ffffff",
    "surface":         "#eceef4",
    "surface_hover":   "#e2e4ee",
    "surface_border":  "#d4d6e2",

    # Text — deep ink, readable hierarchy
    # text_muted bumped from #7a7e96 (3.7:1 fail) to #6b6f88 (4.6:1 WCAG AA)
    "text":            "#181a28",
    "text_secondary":  "#50546e",
    "text_muted":      "#6b6f88",

    # Accent — richer indigo for light backgrounds
    "accent":          "#5b5ef0",
    "accent_hover":    "#4845e0",
    "accent_subtle":   "#eaebff",
    "accent_glow":     "#5b5ef018",

    # Semantic — darker tones for light mode readability
    "success":         "#05875a",
    "success_bg":      "#d4f7e8",
    "warning":         "#c48a05",
    "warning_bg":      "#fef3d0",
    "danger":          "#d42020",
    "danger_bg":       "#fde4e4",

    # Borders & inputs — subtle definition
    "border":          "#cccfe0",
    "border_subtle":   "#dfe1ee",
    "input_bg":        "#ffffff",

    # Scrollbar
    "scrollbar":       "#c0c3d4",
    "scrollbar_hover": "#a4a7be",

    # Tabs
    "tab_active":      "#5b5ef0",
    "tab_inactive":    "#eceef4",

    # Headers
    "header":          "#181a28",

    # Card elevation
    "card_bg":         "#f0f1f7",
    "card_hover":      "#e6e8f0",
    "card_selected":   "#dcdff0",

    # Gradient accents
    "gradient_start":  "#5b5ef0",
    "gradient_end":    "#8b6cf6",

    # Category badge colors
    "badge_model":     "#7c7ff7",
    "badge_lora":      "#05875a",
    "badge_embedding": "#c48a05",
    "badge_text":      "#ffffff",
}

# Active theme state
_current_mode = "dark"
COLORS = dict(_DARK_COLORS)


def set_theme(mode: str):
    """Switch between 'dark' and 'light' themes.

    Updates the global COLORS dict in-place so all style constants
    referencing COLORS stay current.
    """
    global _current_mode, CARD_STYLE, SECTION_HEADER_STYLE, SECTION_SUBHEADER_STYLE
    global ACCENT_BUTTON_STYLE, SUCCESS_BUTTON_STYLE, DANGER_BUTTON_STYLE
    global SECURITY_BANNER_STYLE, MUTED_LABEL_STYLE, STAT_VALUE_STYLE
    global STAT_LABEL_STYLE, TAG_BADGE_STYLE, NAV_BUTTON_STYLE
    global MODEL_CARD_STYLE, SELECTED_CARD_STYLE, TOOLBAR_BUTTON_STYLE
    global SEARCH_INPUT_STYLE, DETAIL_PANEL_STYLE, SIDEBAR_STYLE
    _current_mode = mode
    source = _LIGHT_COLORS if mode == "light" else _DARK_COLORS
    COLORS.clear()
    COLORS.update(source)
    # Re-evaluate style constants so importers see updated colors
    CARD_STYLE = card_style()
    SECTION_HEADER_STYLE = section_header_style()
    SECTION_SUBHEADER_STYLE = section_subheader_style()
    ACCENT_BUTTON_STYLE = accent_button_style()
    SUCCESS_BUTTON_STYLE = success_button_style()
    DANGER_BUTTON_STYLE = danger_button_style()
    SECURITY_BANNER_STYLE = security_banner_style()
    MUTED_LABEL_STYLE = muted_label_style()
    STAT_VALUE_STYLE = stat_value_style()
    STAT_LABEL_STYLE = stat_label_style()
    TAG_BADGE_STYLE = tag_badge_style()
    NAV_BUTTON_STYLE = nav_button_style()
    MODEL_CARD_STYLE = model_card_style()
    SELECTED_CARD_STYLE = selected_card_style()
    TOOLBAR_BUTTON_STYLE = toolbar_button_style()
    SEARCH_INPUT_STYLE = search_input_style()
    DETAIL_PANEL_STYLE = detail_panel_style()
    SIDEBAR_STYLE = sidebar_style()


def get_current_theme() -> str:
    """Return current theme mode ('dark' or 'light')."""
    return _current_mode


def toggle_theme() -> str:
    """Toggle between dark and light. Returns new mode."""
    new_mode = "light" if _current_mode == "dark" else "dark"
    set_theme(new_mode)
    return new_mode


def get_stylesheet() -> str:
    """Return the full Qt stylesheet for the application using current theme colors."""
    c = COLORS
    return f"""
    /* ═══ Foundation ═══
       Sleek splash-aesthetic: cool sci-fi sans-serif, slightly tighter
       letter-spacing, and a subtle 0.2px tracking on body text so the
       UI reads like the floating HUD panels in the splash artwork. */
    QMainWindow, QWidget {{
        background-color: {c['bg']}; color: {c['text']};
        font-family: 'Inter', 'SF Pro Display', 'Segoe UI Variable',
                     'Segoe UI', 'Helvetica Neue', sans-serif;
        font-size: 13px;
        letter-spacing: 0.1px;
    }}
    QLabel {{ color: {c['text']}; background: transparent; }}

    /* ═══ Inputs ═══
       min-height covers the actual widget height (not just content area)
       on macOS where the native style metrics shrink controls below the
       label height — without this the QGridLayout rows collapse and
       inputs overlap labels from neighbouring rows. */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {c['input_bg']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 10px;
        padding: 6px 12px; min-height: 22px;
        selection-background-color: {c['accent']};
        selection-color: white;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1.5px solid {c['accent']};
    }}
    QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
        border-color: {c['surface_border']};
    }}
    QLineEdit::placeholder {{ color: {c['text_muted']}; }}
    QComboBox::drop-down {{
        border: none; padding-right: 12px; width: 28px;
        subcontrol-origin: padding; subcontrol-position: top right;
    }}
    QComboBox::down-arrow {{ width: 10px; height: 10px; }}
    QComboBox QAbstractItemView {{
        background-color: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; selection-background-color: {c['accent']};
        selection-color: white;
        padding: 6px; border-radius: 10px; outline: none;
    }}
    /* Spinbox steppers — transparent so they blend into the input
       instead of looking like a detached panel against input_bg. */
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background-color: transparent; border: none; width: 16px;
        margin-right: 2px;
    }}
    QSpinBox::up-button:hover, QSpinBox::down-button:hover,
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {c['surface_hover']}; border-radius: 3px;
    }}

    /* ═══ Buttons ═══ */
    QPushButton {{
        background-color: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 10px;
        padding: 8px 20px; font-weight: 550; min-height: 24px;
        letter-spacing: 0.2px;
    }}
    QPushButton:hover {{
        background-color: {c['surface_hover']};
        border-color: {c['accent']};
        color: {c['text']};
    }}
    QPushButton:pressed {{
        background-color: {c['accent_subtle']};
        border-color: {c['accent']};
    }}
    QPushButton:focus {{
        border: 1.5px solid {c['accent']};
        outline: none;
    }}
    QPushButton:disabled {{
        color: {c['text_muted']}; background-color: {c['bg_alt']};
        border-color: {c['border_subtle']}; opacity: 0.6;
    }}

    /* ═══ Tables ═══ */
    QTableWidget, QTableView {{
        background-color: {c['bg_alt']}; alternate-background-color: {c['surface']};
        color: {c['text']}; border: 1px solid {c['border']}; border-radius: 12px;
        gridline-color: transparent;
        selection-background-color: {c['accent_subtle']};
        selection-color: {c['text']};
        outline: none;
    }}
    QTableWidget::item, QTableView::item {{
        padding: 6px 12px; border-bottom: 1px solid {c['border_subtle']};
    }}
    QTableWidget::item:selected, QTableView::item:selected {{
        background-color: {c['accent_subtle']};
    }}
    QHeaderView::section {{
        background-color: {c['bg_alt']}; color: {c['text_muted']};
        padding: 10px 12px; border: none;
        border-bottom: 1px solid {c['border']};
        font-weight: 600; font-size: 10px;
        text-transform: uppercase; letter-spacing: 1px;
    }}

    /* ═══ Tabs ═══ */
    QTabWidget::pane {{
        border: 1px solid {c['border']}; border-radius: 12px;
        background-color: {c['bg_alt']}; top: -1px;
    }}
    QTabBar::tab {{
        background-color: transparent; color: {c['text_muted']};
        border: none; border-bottom: 2px solid transparent;
        padding: 11px 24px; margin-right: 2px; font-weight: 500;
        min-width: 64px;
    }}
    QTabBar::tab:selected {{
        color: {c['accent']}; border-bottom: 2.5px solid {c['accent']};
        font-weight: 600;
    }}
    QTabBar::tab:hover:!selected {{
        color: {c['text_secondary']};
        border-bottom: 2px solid {c['border']};
    }}

    /* ═══ Scrollbars ═══ */
    QScrollBar:vertical {{
        background-color: transparent; width: 6px; margin: 6px 1px;
    }}
    QScrollBar::handle:vertical {{
        background-color: {c['scrollbar']}; border-radius: 3px; min-height: 40px;
    }}
    QScrollBar::handle:vertical:hover {{ background-color: {c['scrollbar_hover']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{
        background-color: transparent; height: 6px; margin: 1px 6px;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {c['scrollbar']}; border-radius: 3px; min-width: 40px;
    }}
    QScrollBar::handle:horizontal:hover {{ background-color: {c['scrollbar_hover']}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QAbstractScrollArea::corner {{ background: {c['bg']}; }}

    /* ═══ Progress ═══ */
    QProgressBar {{
        background-color: {c['surface']}; border: none;
        border-radius: 5px; text-align: center; color: {c['text']};
        min-height: 10px; max-height: 10px; font-size: 9px;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {c['gradient_start']}, stop:1 {c['gradient_end']});
        border-radius: 5px;
    }}

    /* ═══ Lists ═══ */
    QListWidget, QListView {{
        background-color: {c['bg_alt']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 10px;
        padding: 6px; outline: none;
    }}
    QListWidget::item, QListView::item {{
        padding: 8px 14px; border-radius: 8px; border: none;
    }}
    QListWidget::item:selected, QListView::item:selected {{
        background-color: {c['accent_subtle']}; color: {c['text']};
    }}
    QListWidget::item:hover:!selected, QListView::item:hover:!selected {{
        background-color: {c['surface_hover']};
    }}

    /* ═══ Text Edit ═══ */
    QTextEdit {{
        background-color: {c['input_bg']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 12px;
        padding: 12px; selection-background-color: {c['accent']};
        selection-color: white;
    }}

    /* ═══ Splitter ═══ */
    QSplitter::handle {{
        background-color: {c['border_subtle']}; width: 1px;
    }}
    QSplitter::handle:hover {{ background-color: {c['accent']}; width: 2px; }}

    /* ═══ Status Bar ═══ */
    QStatusBar {{
        background-color: {c['bg_alt']}; color: {c['text_muted']};
        border-top: 1px solid {c['border_subtle']}; padding: 6px 16px;
        font-size: 11px;
    }}

    /* ═══ Misc ═══ */
    QScrollArea {{ background-color: transparent; border: none; }}
    QDialog {{ background-color: {c['bg']}; border-radius: 12px; }}
    QMessageBox {{ background-color: {c['bg']}; }}
    QMessageBox QLabel {{ color: {c['text']}; }}

    QCheckBox {{ color: {c['text']}; background: transparent; spacing: 8px; }}
    QCheckBox::indicator {{
        width: 18px; height: 18px; border: 1.5px solid {c['border']};
        border-radius: 5px; background-color: {c['input_bg']};
    }}
    QCheckBox::indicator:hover {{ border-color: {c['accent']}; }}
    QCheckBox::indicator:checked {{
        background-color: {c['accent']}; border-color: {c['accent']};
    }}

    /* ═══ Group Box ═══ */
    QGroupBox {{
        background-color: {c['bg_alt']}; border: 1px solid {c['border']};
        border-left: 3px solid {c['accent']}40;
        border-radius: 12px; margin-top: 16px; padding: 18px 14px 14px 14px;
        font-weight: 600; color: {c['text_secondary']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; subcontrol-position: top left;
        padding: 2px 14px; color: {c['text_secondary']};
        font-size: 12px; font-weight: 600;
    }}

    /* ═══ Collapsible Headers ═══ */
    QPushButton[class="collapsible-header"] {{
        background-color: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 10px;
        padding: 10px 16px; font-weight: 600; font-size: 12px;
        text-align: left;
    }}
    QPushButton[class="collapsible-header"]:hover {{
        background-color: {c['surface_hover']}; border-color: {c['accent']};
    }}

    /* ═══ Tooltips ═══ */
    QToolTip {{
        background-color: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 8px;
        padding: 10px 14px; font-size: 12px;
    }}

    /* ═══ Sliders ═══ */
    QSlider::groove:horizontal {{
        background-color: {c['surface']}; height: 4px;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background-color: {c['accent']}; width: 14px; height: 14px;
        margin: -5px 0; border-radius: 7px;
        border: 2px solid {c['bg']};
    }}
    QSlider::handle:horizontal:hover {{
        background-color: {c['accent_hover']}; width: 16px; height: 16px;
        margin: -6px 0; border-radius: 8px;
    }}
    QSlider::sub-page:horizontal {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {c['gradient_start']}, stop:1 {c['gradient_end']});
        border-radius: 2px;
    }}
    QSlider::add-page:horizontal {{
        background-color: {c['surface']}; border-radius: 2px;
    }}
    QSlider::groove:vertical {{
        background-color: {c['surface']}; width: 4px;
        border-radius: 2px;
    }}
    QSlider::handle:vertical {{
        background-color: {c['accent']}; width: 14px; height: 14px;
        margin: 0 -5px; border-radius: 7px;
        border: 2px solid {c['bg']};
    }}
    QSlider::handle:vertical:hover {{
        background-color: {c['accent_hover']};
    }}

    /* ═══ Context Menus ═══ */
    QMenu {{
        background-color: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 10px;
        padding: 6px 0;
    }}
    QMenu::item {{
        padding: 8px 32px 8px 18px; border-radius: 6px;
        margin: 2px 6px;
    }}
    QMenu::item:selected {{
        background-color: {c['accent_subtle']}; color: {c['accent']};
    }}
    QMenu::separator {{
        height: 1px; background-color: {c['border_subtle']};
        margin: 6px 12px;
    }}
    QMenu::icon {{ padding-left: 10px; }}
    """


# --- Reusable style functions (re-evaluated on theme change) ---

def card_style() -> str:
    """Card containers with layered depth."""
    return (
        f"background-color: {COLORS['bg_alt']}; "
        f"border: 1px solid {COLORS['border']}; "
        f"border-radius: 14px; padding: 16px;"
    )

def section_header_style() -> str:
    """Primary section header labels."""
    return (
        f"color: {COLORS['header']}; font-size: 15px; font-weight: 700; "
        f"padding: 6px 0 2px 0; background: transparent; letter-spacing: 0.2px;"
    )

def section_subheader_style() -> str:
    """Secondary section headers — smaller, uppercase, muted."""
    return (
        f"color: {COLORS['text_muted']}; font-size: 10px; font-weight: 700; "
        f"padding: 2px 0; background: transparent; text-transform: uppercase; "
        f"letter-spacing: 1.2px;"
    )

def accent_button_style() -> str:
    """Primary action buttons with gradient accent."""
    return (
        f"QPushButton {{ "
        f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
        f"stop:0 {COLORS['gradient_start']}, stop:1 {COLORS['gradient_end']}); "
        f"color: white; border: none; border-radius: 10px; "
        f"padding: 10px 24px; font-weight: 650; font-size: 13px; "
        f"letter-spacing: 0.3px; }} "
        f"QPushButton:hover {{ "
        f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
        f"stop:0 {COLORS['accent_hover']}, stop:1 {COLORS['gradient_end']}); }} "
        f"QPushButton:pressed {{ background-color: {COLORS['accent']}; }} "
        f"QPushButton:disabled {{ background-color: {COLORS['surface']}; "
        f"color: {COLORS['text_muted']}; border: 1px solid {COLORS['border']}; }}"
    )

def success_button_style() -> str:
    """Success/confirmation buttons."""
    return (
        f"QPushButton {{ background-color: {COLORS['success']}; color: {COLORS['bg']}; "
        f"border: none; border-radius: 10px; padding: 10px 24px; "
        f"font-weight: 700; font-size: 13px; }} "
        f"QPushButton:hover {{ background-color: {COLORS['accent']}; color: white; }} "
        f"QPushButton:disabled {{ background-color: {COLORS['surface']}; "
        f"color: {COLORS['text_muted']}; border: 1px solid {COLORS['border']}; }}"
    )

def danger_button_style() -> str:
    """Destructive action buttons — outlined danger."""
    return (
        f"QPushButton {{ background-color: transparent; color: {COLORS['danger']}; "
        f"border: 1.5px solid {COLORS['danger']}; border-radius: 10px; "
        f"padding: 9px 20px; font-weight: 600; }} "
        f"QPushButton:hover {{ background-color: {COLORS['danger_bg']}; "
        f"border-color: {COLORS['danger']}; }} "
        f"QPushButton:pressed {{ background-color: {COLORS['danger']}; color: white; }}"
    )

def security_banner_style() -> str:
    """Status banners with accent left border."""
    return (
        f"background-color: {COLORS['success_bg']}; color: {COLORS['success']}; "
        f"border: 1px solid {COLORS['success']}; border-radius: 12px; "
        f"border-left: 4px solid {COLORS['success']}; "
        f"padding: 12px 20px; font-weight: 500; font-size: 12px;"
    )

def muted_label_style() -> str:
    """De-emphasized labels."""
    return (
        f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent;"
    )

def stat_value_style() -> str:
    """Large statistic values in accent color."""
    return (
        f"color: {COLORS['accent']}; font-size: 24px; font-weight: 800; "
        f"background: transparent; letter-spacing: -0.5px;"
    )

def stat_label_style() -> str:
    """Tiny uppercase stat labels."""
    return (
        f"color: {COLORS['text_muted']}; font-size: 10px; font-weight: 700; "
        f"background: transparent; text-transform: uppercase; letter-spacing: 1px;"
    )

def tag_badge_style() -> str:
    """Tag/badge chips — pill shape with accent tint."""
    return (
        f"background-color: {COLORS['accent_subtle']}; color: {COLORS['accent']}; "
        f"border: 1px solid {COLORS['accent']}40; "
        f"border-radius: 12px; padding: 3px 14px; "
        f"font-size: 11px; font-weight: 600;"
    )

def nav_button_style() -> str:
    """Navigation buttons with hover highlight."""
    return (
        f"QPushButton {{ background-color: {COLORS['surface']}; color: {COLORS['text']}; "
        f"border: 1px solid {COLORS['border']}; border-radius: 10px; "
        f"padding: 8px 22px; font-weight: 550; }} "
        f"QPushButton:hover {{ background-color: {COLORS['surface_hover']}; "
        f"border-color: {COLORS['accent']}; color: {COLORS['text']}; }} "
        f"QPushButton:disabled {{ color: {COLORS['text_muted']}; "
        f"background-color: {COLORS['bg']}; border-color: {COLORS['border_subtle']}; }}"
    )

def model_card_style() -> str:
    """Library model cards."""
    return (
        f"background-color: {COLORS['card_bg']}; "
        f"border: 1px solid {COLORS['border']}; "
        f"border-radius: 14px; padding: 14px;"
    )

def selected_card_style() -> str:
    """Selected/active card with accent border."""
    return (
        f"background-color: {COLORS['card_selected']}; "
        f"border: 2px solid {COLORS['accent']}; "
        f"border-radius: 14px; padding: 13px;"
    )

def category_badge_style(category: str) -> str:
    """Badge style for model/lora/embedding categories."""
    color_key = {
        "model": "badge_model",
        "lora": "badge_lora",
        "embedding": "badge_embedding",
    }.get(category, "badge_model")
    return (
        f"background-color: {COLORS[color_key]}; "
        f"color: {COLORS['badge_text']}; "
        f"border-radius: 12px; padding: 3px 12px; "
        f"font-size: 11px; font-weight: 700;"
    )

def toolbar_button_style() -> str:
    """Compact toolbar buttons."""
    return (
        f"QPushButton {{ background-color: {COLORS['surface']}; color: {COLORS['text']}; "
        f"border: 1px solid {COLORS['border']}; border-radius: 8px; "
        f"padding: 5px 12px; font-weight: 550; font-size: 12px; }} "
        f"QPushButton:hover {{ background-color: {COLORS['surface_hover']}; "
        f"border-color: {COLORS['accent']}; }} "
        f"QPushButton:pressed {{ background-color: {COLORS['accent_subtle']}; }}"
    )

def search_input_style() -> str:
    """Larger search input."""
    return (
        f"background-color: {COLORS['input_bg']}; color: {COLORS['text']}; "
        f"border: 1px solid {COLORS['border']}; border-radius: 12px; "
        f"padding: 10px 16px 10px 36px; font-size: 14px; min-height: 28px;"
    )

def detail_panel_style() -> str:
    """Detail/info panels at bottom of views."""
    return (
        f"background-color: {COLORS['bg_alt']}; "
        f"border: 1px solid {COLORS['border']}; "
        f"border-top: 3px solid {COLORS['accent']}30; "
        f"border-radius: 0 0 12px 12px; padding: 16px;"
    )

def sidebar_style() -> str:
    """Navigation sidebar buttons."""
    return (
        f"QPushButton {{ background-color: transparent; color: {COLORS['text_secondary']}; "
        f"border: none; border-radius: 10px; "
        f"padding: 10px 18px; font-weight: 500; text-align: left; }} "
        f"QPushButton:hover {{ background-color: {COLORS['surface']}; "
        f"color: {COLORS['text']}; }} "
        f"QPushButton:checked {{ background-color: {COLORS['accent_subtle']}; "
        f"color: {COLORS['accent']}; font-weight: 600; }}"
    )


# --- Backward-compatible constants (evaluated at import time, dark theme) ---
CARD_STYLE = card_style()
SECTION_HEADER_STYLE = section_header_style()
SECTION_SUBHEADER_STYLE = section_subheader_style()
ACCENT_BUTTON_STYLE = accent_button_style()
SUCCESS_BUTTON_STYLE = success_button_style()
DANGER_BUTTON_STYLE = danger_button_style()
SECURITY_BANNER_STYLE = security_banner_style()
MUTED_LABEL_STYLE = muted_label_style()
STAT_VALUE_STYLE = stat_value_style()
STAT_LABEL_STYLE = stat_label_style()
TAG_BADGE_STYLE = tag_badge_style()
NAV_BUTTON_STYLE = nav_button_style()
MODEL_CARD_STYLE = model_card_style()
SELECTED_CARD_STYLE = selected_card_style()
TOOLBAR_BUTTON_STYLE = toolbar_button_style()
SEARCH_INPUT_STYLE = search_input_style()
DETAIL_PANEL_STYLE = detail_panel_style()
SIDEBAR_STYLE = sidebar_style()
