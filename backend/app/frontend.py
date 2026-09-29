import gradio as gr

from api_client import stream_llama,ingest_pdf

#fixed set of models
MODELS = [
    (
        "Qwen 3.5 9B",
        "unsloth/Qwen3.5-9B-GGUF",
    ),
    (
        "Qwen 3.5 4B",
        "unsloth/Qwen3.5-4B-GGUF",
    ),
    (
        "Ministral 3 8B Instruct",
        "mistralai/Ministral-3-8B-Instruct-2512-GGUF",
    ),
    (
        "Ministral 3 8B Reasoning",
        "mistralai/Ministral-3-8B-Reasoning-2512-GGUF",
    ),
    (
        "Gemma 4 4B",
        "unsloth/gemma-4-4B-it-GGUF",
    ),
    (
        "Ministral 3 3B Instruct",
        "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
    ),
    (
        "Granite 4.1 3B",
        "ibm-granite/granite-4.1-3b-GGUF",
    ),
    (
        "Phi-4 Mini Instruct",
        "unsloth/Phi-4-mini-instruct-GGUF",
    ),
    (
        "Phi-4 Mini Reasoning",
        "unsloth/Phi-4-mini-reasoning-GGUF",
    ),
    (
        "Ministral 3 14B Instruct",
        "mistralai/Ministral-3-14B-Instruct-2512-GGUF",
    ),
]


# ============================================================
# Demo chat data
# UI only for now
# ============================================================

DEMO_CHATS = [
    ("Python debugging", "chat-1"),
    ("Project architecture", "chat-2"),
    ("Ideas for the app", "chat-3"),
    ("FastAPI questions", "chat-4"),
]


#callbacks

def send_message(message, history):
    """
    UI callback.

    Sends the user's message to the backend API
    and progressively updates the Gradio chatbot.
    """

    if not message or not message.strip():
        yield history, ""
        return

    history = history or []

    # Add user message
    history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    # Add empty assistant message
    history.append(
        {
            "role": "assistant",
            "content": "",
        }
    )

    # Immediately update the UI
    yield history, ""

    response_text = ""

    try:
        for chunk in stream_llama(message):

            response_text += chunk

            history[-1]["content"] = response_text

            yield history, ""

    except Exception as exc:

        history[-1]["content"] = (
            f"Backend error: {exc}"
        )

        yield history, ""

def handle_pdf_upload(file_path):

    if not file_path:
        return

    try:

        result = ingest_pdf(
            file_path
        )

        gr.Info(
            f"Indexed {result['filename']} "
            f"({result['chunks']} chunks)"
        )

    except Exception as exc:

        raise gr.Error(
            f"PDF ingestion failed: {exc}"
        )

def new_chat():
    """
    Placeholder for future chat creation logic.
    """
    return []


def select_chat(chat_id):
    """
    Placeholder for future chat loading logic.
    """
    if not chat_id:
        return []

    return []


def search_chats(query):
    """
    UI-only chat filtering.
    """
    query = (query or "").strip().lower()

    if not query:
        return gr.update(
            choices=DEMO_CHATS,
            value=None,
        )

    filtered = [
        (title, value)
        for title, value in DEMO_CHATS
        if query in title.lower()
    ]

    return gr.update(
        choices=filtered,
        value=None,
    )


# ============================================================
# App
# ============================================================

with gr.Blocks(
    title="Local AI",
) as demo:

    # ========================================================
    # Application shell
    # ========================================================

    with gr.Row(
        elem_id="app-shell",
        equal_height=False,
    ):

        # ====================================================
        # SIDEBAR
        # ====================================================

        with gr.Column(
            elem_id="sidebar",
            scale=1,
            min_width=280,
        ):

            # ------------------------------------------------
            # Brand
            # ------------------------------------------------

            gr.HTML(
                """
                <div class="brand">
                    <div class="brand-mark">✦</div>

                    <div>
                        <div class="brand-name">
                            Local AI
                        </div>

                        <div class="brand-caption">
                            Private · Local · Yours
                        </div>
                    </div>
                </div>
                """,
                elem_id="brand",
            )

            # ------------------------------------------------
            # New Chat
            # ------------------------------------------------

            new_chat_button = gr.Button(
                "＋  New chat",
                variant="primary",
                elem_id="new-chat-button",
            )

            # ------------------------------------------------
            # Chats
            # ------------------------------------------------

            gr.Markdown(
                "CHATS",
                elem_classes=["section-label"],
            )

            chat_list = gr.Dropdown(
                choices=DEMO_CHATS,
                value=None,
                show_label=False,
                container=False,
                elem_id="chat-list",
            )

            # ------------------------------------------------
            # Spacer
            # ------------------------------------------------

            gr.HTML(
                '<div class="sidebar-spacer"></div>'
            )

            # ------------------------------------------------
            # Model Library
            # ------------------------------------------------
            model_selector = gr.Dropdown(
                    choices=MODELS,
                    value=MODELS[0][1],
                    show_label=False,
                    container=False,
                    interactive=True,
                    filterable=False,
                    elem_id="model-dropdown",
                )
            # ------------------------------------------------
            # Sidebar footer
            # ------------------------------------------------

            with gr.Row(
                elem_id="sidebar-footer",
            ):

                gr.HTML(
                    """
                    <div class="footer-status">
                        <span class="status-dot"></span>
                        Local environment
                    </div>
                    """
                )

        # ====================================================
        # MAIN AREA
        # ====================================================

        with gr.Column(
            elem_id="main-area",
            scale=4,
        ):

            # ------------------------------------------------
            # Header
            # ------------------------------------------------

            with gr.Row(
                elem_id="topbar",
                equal_height=True,
            ):

                gr.HTML(
                    """
                    <div class="page-heading">
                        <div class="page-title">
                            Chat
                        </div>

                        <div class="page-subtitle">
                            Talk to your local models
                        </div>
                    </div>
                    """
                )


            # ------------------------------------------------
            # Chat
            # ------------------------------------------------

            chatbot = gr.Chatbot(
                label="",
                height=650,
                elem_id="chatbot",
                show_label=False,
            )

            # ------------------------------------------------
            # Composer
            # ------------------------------------------------

            with gr.Row(
                elem_id="composer",
                equal_height=True,
            ):
                pdf_upload = gr.UploadButton(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    file_count="single",
                    elem_id="pdf-upload",
                    scale=0,
                )

                message = gr.Textbox(
                    placeholder="Message your model...",
                    show_label=False,
                    lines=1,
                    max_lines=8,
                    elem_id="message-input",
                    container=False,
                    scale=9,
                )

                send_button = gr.Button(
                    "↑",
                    variant="primary",
                    elem_id="send-button",
                    scale=0,
                    min_width=52,
                )

            gr.HTML(
                """
                <div class="composer-hint">
                    Enter to send · Shift + Enter for a new line
                </div>
                """,
                elem_id="composer-hint",
            )

    # ========================================================
    # Events
    # ========================================================

    # New chat
    new_chat_button.click(
        fn=new_chat,
        inputs=None,
        outputs=chatbot,
    )

    # Select existing chat
    chat_list.change(
        fn=select_chat,
        inputs=chat_list,
        outputs=chatbot,
    )

    pdf_upload.upload(
    fn=handle_pdf_upload,
    inputs=pdf_upload,
    outputs=None,
    )

   


    # Keep the header model display synchronized
    # with the sidebar model selector.
    # model_selector.change(
    #     fn=lambda model: model,
    #     inputs=model_selector,
    #     outputs=selected_model_display,
    # )

    # --------------------------------------------------------
    # Chat submit placeholders
    #
    # These intentionally do nothing yet.
    # We will connect them to your FastAPI streaming
    # endpoint afterward.
    # --------------------------------------------------------

    message.submit(
        fn=send_message,
        inputs=[
            message,
            chatbot,
        ],
        outputs=[
            chatbot,
            message,
        ],
    )

    send_button.click(
        fn=send_message,
        inputs=[
            message,
            chatbot,
        ],
        outputs=[
            chatbot,
            message,
        ],
    )
# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(
            primary_hue="cyan",
            neutral_hue="slate",
        ),
        css_paths=["styles.css"],
    )