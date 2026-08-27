"""AI Spirit shared Gradio theme — dark, modern, emerald accent."""

from __future__ import annotations

import gradio as gr

SPIRIT_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    # Background
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    # Blocks
    block_background_fill="#1e293b",
    block_background_fill_dark="#1e293b",
    block_border_width="1px",
    block_border_color="#334155",
    block_border_color_dark="#334155",
    block_label_text_color="#94a3b8",
    block_label_text_color_dark="#94a3b8",
    block_title_text_color="#e2e8f0",
    block_title_text_color_dark="#e2e8f0",
    block_shadow="0 4px 6px -1px rgba(0,0,0,0.3)",
    block_shadow_dark="0 4px 6px -1px rgba(0,0,0,0.3)",
    block_radius="12px",
    # Inputs
    input_background_fill="#0f172a",
    input_background_fill_dark="#0f172a",
    input_border_color="#334155",
    input_border_color_dark="#334155",
    input_border_width="1px",
    input_radius="8px",
    # Buttons
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_dark="*primary_600",
    button_primary_background_fill_hover="*primary_500",
    button_primary_background_fill_hover_dark="*primary_500",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_secondary_background_fill="#334155",
    button_secondary_background_fill_dark="#334155",
    button_secondary_text_color="#e2e8f0",
    button_secondary_text_color_dark="#e2e8f0",
    button_cancel_background_fill="#dc2626",
    button_cancel_background_fill_dark="#dc2626",
    button_cancel_text_color="white",
    button_cancel_text_color_dark="white",
    button_large_radius="10px",
    button_small_radius="8px",
    # Text
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    body_text_color_subdued="#94a3b8",
    body_text_color_subdued_dark="#94a3b8",
    # Tabs
    # Borders
    border_color_primary="#334155",
    border_color_primary_dark="#334155",
    # Shadows
    shadow_spread="0px",
)
