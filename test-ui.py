import gradio as gr

def respond(message, history):
    # your model call here
    return f"You said: {message}"


with gr.Blocks(fill_height=True) as demo:

    chatbot = gr.Chatbot(
        height=650,
        show_label=False,
        layout="panel",
    )

    textbox = gr.Textbox(
        placeholder="Message your model...",
        show_label=False,
        lines=1,
        max_lines=8,
        container=False,
    )

    gr.ChatInterface(
        fn=respond,
        chatbot=chatbot,
        textbox=textbox,

        # built directly into the composer
        submit_btn=True,
        stop_btn=True,

        fill_height=True,
    )

demo.launch()