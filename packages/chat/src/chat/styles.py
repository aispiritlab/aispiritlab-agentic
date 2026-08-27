"""AI Spirit shared CSS — workspace tabs, buttons, animations, cards."""

from __future__ import annotations

WORKSPACE_TABS_CSS = """
.workspace-tabs {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px 16px;
    background: #0a0f1a;
    border-bottom: 2px solid #1e293b;
    overflow-x: auto;
    scrollbar-width: none;
}
.workspace-tabs::-webkit-scrollbar { display: none; }
.workspace-tab {
    padding: 8px 20px;
    border-radius: 8px 8px 0 0;
    background: #1e293b;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.2s ease;
    font-weight: 500;
    font-size: 13px;
    white-space: nowrap;
    border: 1px solid transparent;
    border-bottom: none;
    user-select: none;
}
.workspace-tab:hover {
    background: #334155;
    color: #e2e8f0;
}
.workspace-tab.active {
    background: #1e293b;
    color: #f8fafc;
    border-color: #334155;
    box-shadow: 0 -2px 0 #10b981 inset;
    font-weight: 600;
}
.workspace-tab-editor {
    background: #0f172a;
    border: 1px solid #10b981;
    border-bottom: none;
    color: #10b981;
    font-weight: 600;
}
.workspace-tab-editor.active {
    background: #064e3b;
    color: #34d399;
    box-shadow: 0 -2px 0 #10b981 inset;
}
.workspace-tab-new {
    background: transparent;
    border: 1px dashed #475569;
    color: #64748b;
    font-size: 16px;
    padding: 6px 14px;
}
.workspace-tab-new:hover {
    border-color: #10b981;
    color: #10b981;
}
"""

BUTTON_CSS = """
.spirit-btn {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px;
}
.spirit-btn:active { transform: scale(0.97); }
button.primary {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    box-shadow: 0 2px 8px rgba(16,185,129,0.25) !important;
    border: none !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(16,185,129,0.35) !important;
}
button.stop {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    box-shadow: 0 2px 8px rgba(239,68,68,0.25) !important;
}
"""

ANIMATION_CSS = """
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 8px rgba(16,185,129,0.3); }
    50% { box-shadow: 0 0 24px rgba(16,185,129,0.6); }
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.running-animation {
    animation: pulse-glow 1.5s ease-in-out infinite;
}
.fade-in {
    animation: fadeIn 0.3s ease-out;
}
"""

CARD_CSS = """
.settings-card {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin: 6px 0 !important;
}
.settings-card-header {
    font-size: 11px;
    font-weight: 700;
    color: #10b981;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #334155;
}
"""

CONTEXT_BAR_CSS = """
.context-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 16px;
    background: #1e293b;
    border-bottom: 1px solid #334155;
    font-size: 13px;
    color: #94a3b8;
}
.context-bar-label {
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.5px;
}
"""

HEADER_CSS = """
.spirit-header {
    padding: 12px 20px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-bottom: 1px solid #334155;
}
.spirit-header h1 {
    font-size: 20px !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #10b981, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
}
"""

GLOBAL_CSS = f"{WORKSPACE_TABS_CSS}\n{BUTTON_CSS}\n{ANIMATION_CSS}\n{CARD_CSS}\n{CONTEXT_BAR_CSS}\n{HEADER_CSS}"
