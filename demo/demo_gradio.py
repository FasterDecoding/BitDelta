import gradio as gr
import random
import time
import requests
import functools

import json

url = "http://localhost:8000"
models = requests.get(f"{url}/models").json()
layouts = {
    6: (2, 3)
}
num_models = len(models)

max_new_tokens = 200

with gr.Blocks() as demo:
    assert len(models) in layouts, f"Layout for {len(models)} models not found"
    layout = layouts[len(models)]
    chatbots = []
    check_buttons = []
    titles = []
    tot = 0
    gr.HTML("""<h1 align="center">🔥BitDelta Demo🔥</h1>""")
    gr.HTML(f"""
    <p align="center">We are using less than 30GB of GPU memory to run 6 Mistral-7B models at the same time!</p>
    <p align="center">You can choose the reply you like the most. All 6 fine-tunes will continue the chat history from the chosen reply. </p>
    <p align="center">We currently truncate each reply to {max_new_tokens} tokens, but you can modify this in <code>demo_gradio.py</p>""")
    for i in range(layout[0]):
        with gr.Row():
            for j in range(layout[1]):
                with gr.Column():
                    title = gr.HTML(f"<h3>{models[tot]}</h3>")
                    titles.append(title)
                    chatbot = gr.Chatbot(height="320px")
                    chatbots.append(chatbot)
                    check_button = gr.Button("Choose this reply")
                    check_buttons.append(check_button)
                    tot += 1

    # Default checkbox
    chosen = gr.State(0)
    chat_history = gr.State([])
    check_buttons[0].value = "✅Chosen✅"

    msg = gr.Textbox()
    clear = gr.ClearButton([msg] + chatbots)

    clear.click(lambda: [0, []], [], [chosen, chat_history])

    def respond(message, chat_history, chosen):
        # Convert chat_history to openai format
        messages = []
        for i in range(len(chat_history)):
            messages.append({
                "role": "user",
                "content": chat_history[i][0]
            })
            messages.append({
                "role": "assistant",
                "content": chat_history[i][1]
            })
        messages.append({
            "role": "user",
            "content": message
        })

        response = [""] * num_models
        
        # set all_chat_history to all the chosen chat_history at the beginning
        all_chat_history = [chat_history + [(message, "")] for i in range(num_models)]
        chosen_chat_history = chat_history + [(message, "")]

        yield "", chosen_chat_history, *all_chat_history

        # streaming imp
        stream_response = requests.post(f"{url}/generate?max_new_tokens={max_new_tokens}", json=messages, stream=True)
        if stream_response.status_code != 200:
            print(f"Error: {stream_response.status_code}")
            return

        try:
            for line in stream_response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    # Process each line as needed
                    token_response = json.loads(decoded_line)
                    # print(f"TOKEN RESPONSE: {token_response}")

                    # quick hack
                    # for i in range(num_models):
                    #     if token_response["response"][i] in ["<|end_of_turn|>", "<|im_end|>", "</s>"]:
                    #         token_response["response"][i] = ""

                    # response = [response[i] + token_response["response"][i] for i in range(num_models)]
                    continue_gen = [token_response["response"][i][1] == "continue" for i in range(num_models)]
                    response = [token_response["response"][i][0] if continue_gen[i] else response[i] for i in range(num_models)]
                    all_chat_history = [chat_history + [(message, response[i])] for i in range(num_models)]
                    chosen_chat_history = chat_history + [(message, response[chosen])]

                    yield "", chosen_chat_history, *all_chat_history
        
        except Exception as e:
            print(f"Exception: {e}")
            return
    
    msg.submit(respond, [msg, chat_history, chosen], [msg, chat_history] + chatbots)

    def check(i, chat_history, chatbot, chosen):
        chosen = i
        chat_history = chatbot
        new_labels = []
        for j in range(len(chatbots)):
            if j == i:
                new_labels.append("✅Chosen✅")
            else:
                new_labels.append("Choose this reply")
        return chat_history, chosen, *new_labels
    
    for i in range(len(chatbots)):
        check_buttons[i].click(functools.partial(check, i), [chat_history, chatbots[i], chosen], [chat_history, chosen] + check_buttons)


demo.queue(max_size=10)
demo.launch(share=True)
