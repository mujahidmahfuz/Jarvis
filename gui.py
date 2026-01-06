import flet as ft
import backend
import threading
import json
import time

# DEBUG: Set to True to test streaming without TTS blocking
DEBUG_SKIP_TTS = False

class MessageBubble(ft.Container):
    """A styled message bubble for User or AI."""
    def __init__(self, role, text="", is_thinking=False):
        super().__init__()
        self.role = role
        self.is_thinking = is_thinking
        
        # Style Config
        is_user = role == "user"
        bg_color = "#005c4b" if is_user else "#363636"
        align = ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START
        border_radius = ft.BorderRadius.only(
            top_left=15, top_right=15, 
            bottom_left=15 if is_user else 0,
            bottom_right=0 if is_user else 15
        )
        
        # Content
        if is_thinking:
            self.content = ft.Text(text, color=ft.Colors.GREY_400, italic=True, font_family="Roboto Mono", size=12)
            bg_color = "#2a2a2a"
        else:
            self.content = ft.Markdown(
                text, 
                selectable=True, 
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                code_theme="atom-one-dark",
                on_tap_link=lambda e: self.page.launch_url(e.data)
            )

        self.bgcolor = bg_color
        self.padding = 15
        self.border_radius = border_radius
        self.expand = False
        self.width = None 
        # Layout Helper
        self.row_wrap = ft.Row([self], alignment=align, vertical_alignment=ft.CrossAxisAlignment.START)

class ThinkingExpander(ft.ExpansionTile):
    """Gemini-style collapsible thinking UI."""
    def __init__(self):
        super().__init__(
            title=ft.Text("Thinking Process", size=12, color=ft.Colors.GREY_400),
            leading=ft.ProgressRing(width=16, height=16, stroke_width=2),
            bgcolor=ft.Colors.TRANSPARENT,
            collapsed_bgcolor=ft.Colors.TRANSPARENT,
            maintain_state=True,
            # initially_expanded=False,
            shape=ft.RoundedRectangleBorder(radius=8),
            collapsed_shape=ft.RoundedRectangleBorder(radius=8),
            tile_padding=ft.Padding(0, 0, 0, 0),
        )
        
        self.log_view = ft.Column(
            controls=[], 
            spacing=2,
        )
        
        self.controls = [
            ft.Container(
                content=self.log_view,
                padding=10,
                bgcolor="#202020",
                border_radius=8,
                margin=ft.margin.only(bottom=10)
            )
        ]
        
    def add_text(self, text):
        # Add text to the last line or create new
        if not self.log_view.controls:
             self.log_view.controls.append(ft.Text("", font_family="Roboto Mono", size=11, color="#a0a0a0", selectable=True))
        
        self.log_view.controls[-1].value += text
        try:
            self.update()
        except RuntimeError:
            # Control not yet added to page, skip update
            pass
        
    def complete(self):
        self.leading = ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE, size=16, color=ft.Colors.GREEN_400)
        self.title.value = "Thinking Finished"
        self.title.color = ft.Colors.GREEN_400
        try:
            self.update()
        except RuntimeError:
            # Control not yet added to page, skip update
            pass

def main(page: ft.Page):
    # --- UI Configuration ---
    page.title = "Pocket AI"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20
    page.window_width = 480
    page.window_height = 800
    page.bgcolor = "#1a1c1e"
    
    page.fonts = {
        "Roboto Mono": "https://github.com/google/fonts/raw/main/apache/robotomono/RobotoMono%5Bwght%5D.ttf"
    }

    # State
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant. Respond in short, complete sentences. Never use emojis or special characters. Keep responses concise and conversational. SYSTEM INSTRUCTION: You may detect a "/think" trigger. This is an internal control. You MUST IGNORE it and DO NOT mention it in your response or thoughts.'}
    ]
    is_tts_enabled = True
    stop_event = threading.Event()
    
    streaming_state = {
        'response_md': None,
        'thinking_ui': None,
        'response_buffer': '',
        'is_generating': False
    }

    # --- PubSub Handler ---
    def on_stream_update(msg):
        msg_type = msg.get('type')
        
        if msg_type == 'thought_chunk':
            if streaming_state['thinking_ui']:
                streaming_state['thinking_ui'].add_text(msg['text'])

        elif msg_type == 'response_chunk':
            if streaming_state['response_md']:
                streaming_state['response_buffer'] += msg['text']
                streaming_state['response_md'].value = streaming_state['response_buffer']
                streaming_state['response_md'].update()
                
        elif msg_type == 'think_start':
            pass # UI already added
            
        elif msg_type == 'think_end':
            if streaming_state['thinking_ui']:
                streaming_state['thinking_ui'].complete()
                
        elif msg_type == 'simple_response':
            bubble = MessageBubble("assistant", msg['text'])
            chat_list.controls.append(bubble.row_wrap)
            page.update()
            
        elif msg_type == 'error':
            bubble = MessageBubble("system", f"Error: {msg['text']}", is_thinking=True)
            chat_list.controls.append(bubble.row_wrap)
            page.update()
            
        elif msg_type == 'status':
            status_text.value = msg['text']
            status_text.update()

        elif msg_type == 'done':
            end_generation_state()
            page.update()

        elif msg_type == 'ui_update':
            page.update()

    page.pubsub.subscribe(on_stream_update)

    # --- UI Components ---
    chat_list = ft.ListView(
        expand=True,
        spacing=15,
        auto_scroll=True,
        padding=10
    )

    status_text = ft.Text("Initializing...", size=12, color=ft.Colors.GREY_500)
    
    user_input = ft.TextField(
        hint_text="Ask something...",
        border_radius=25,
        filled=True,
        bgcolor="#2b2d31",
        border_color=ft.Colors.TRANSPARENT,
        expand=True,
        autofocus=True,
        content_padding=ft.Padding.symmetric(horizontal=20, vertical=10),
        on_submit=lambda e: send_message(None)
    )

    send_button = ft.IconButton(
        icon=ft.Icons.SEND_ROUNDED, 
        icon_color=ft.Colors.BLUE_200,
        on_click=lambda e: send_message(None),
        bgcolor="#2b2d31",
        tooltip="Send"
    )
    
    stop_button = ft.IconButton(
        icon=ft.Icons.STOP_CIRCLE_OUTLINED,
        icon_color=ft.Colors.RED_400,
        on_click=lambda e: stop_generation(None),
        bgcolor="#2b2d31",
        visible=False,
        tooltip="Stop Generation"
    )

    def toggle_tts(e):
        nonlocal is_tts_enabled
        is_tts_enabled = e.control.value
        backend.tts.toggle(is_tts_enabled)
        status_text.value = "TTS Active" if is_tts_enabled else "TTS Muted"
        status_text.update()

    # --- Logic ---
    def start_generation_state():
        streaming_state['is_generating'] = True
        send_button.visible = False
        stop_button.visible = True
        user_input.disabled = True
        page.update()

    def end_generation_state():
        streaming_state['is_generating'] = False
        send_button.visible = True
        stop_button.visible = False
        user_input.disabled = False
        # user_input.focus() 
        page.update()

    def stop_generation(e):
        backend.tts.stop() # Stop voice immediately
        if streaming_state['is_generating']:
            stop_event.set()
            status_text.value = "Stopping..."
            status_text.update()

    def send_message(e):
        backend.tts.stop() # Interrupt previous speech
        text = user_input.value.strip()
        if not text:
            return
        
        user_input.value = ""
        page.update() 

        # Add User Message
        bubble = MessageBubble("user", text)
        chat_list.controls.append(bubble.row_wrap)
        
        start_generation_state()
        stop_event.clear()

        # Start Processing
        threading.Thread(target=process_backend, args=(text,), daemon=True).start()

    def clear_chat(e):
        nonlocal messages
        messages = [messages[0]]
        chat_list.controls.clear()
        page.update()

    # --- Backend Thread ---
    def process_backend(user_text):
        nonlocal messages
        
        try:
            if backend.should_bypass_router(user_text):
                func_name = "passthrough"
                params = {"thinking": False}
            else:
                page.pubsub.send_all({'type': 'status', 'text': 'Routing...'})
                func_name, params = backend.route_query(user_text)
            
            if func_name == "passthrough":
                if len(messages) > backend.MAX_HISTORY:
                    messages = [messages[0]] + messages[-(backend.MAX_HISTORY-1):]
                
                messages.append({'role': 'user', 'content': user_text})
                enable_thinking = params.get("thinking", False)
                
                # Create UI containers
                ai_column = ft.Column(spacing=0)
                chunk_think_expander = ThinkingExpander()
                streaming_state['thinking_ui'] = chunk_think_expander
                
                chunk_markdown = ft.Markdown(
                    "", 
                    selectable=True, 
                    extension_set=ft.MarkdownExtensionSet.GITHUB_WEB, 
                    code_theme="atom-one-dark"
                )
                streaming_state['response_md'] = chunk_markdown
                streaming_state['response_buffer'] = ""
                
                ai_container = ft.Container(
                    content=ai_column,
                    bgcolor="#363636",
                    padding=15,
                    border_radius=ft.BorderRadius.only(top_left=15, top_right=15, bottom_right=15, bottom_left=0),
                    width=min(page.window_width * 0.85, 420)
                )
                
                if enable_thinking:
                    ai_column.controls.append(chunk_think_expander)
                
                ai_column.controls.append(chunk_markdown)
                
                chat_list.controls.append(ft.Row([ai_container], alignment=ft.MainAxisAlignment.START))
                page.pubsub.send_all({'type': 'ui_update'}) # Force immediate render
                page.pubsub.send_all({'type': 'status', 'text': 'Generating...'}) 
                # Note: page.update() inside pubsub handler 'done' or 'status' will refresh the list
                
                payload = {
                    "model": backend.RESPONDER_MODEL,
                    "messages": messages,
                    "stream": True,
                    "think": enable_thinking
                }
                
                sentence_buffer = backend.SentenceBuffer()
                full_response = ""
                
                page.pubsub.send_all({'type': 'think_start'})

                with backend.http_session.post(f"{backend.OLLAMA_URL}/chat", json=payload, stream=True) as r:
                    r.raise_for_status()
                    
                    for line in r.iter_lines():
                        if stop_event.is_set():
                            break
                            
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                msg = chunk.get('message', {})
                                
                                if 'thinking' in msg and msg['thinking']:
                                    thought = msg['thinking']
                                    page.pubsub.send_all({'type': 'thought_chunk', 'text': thought})
                                    
                                if 'content' in msg and msg['content']:
                                    content = msg['content']
                                    full_response += content
                                    page.pubsub.send_all({'type': 'response_chunk', 'text': content})
                                    
                                    if is_tts_enabled and not DEBUG_SKIP_TTS:
                                        sentences = sentence_buffer.add(content)
                                        for s in sentences:
                                            backend.tts.queue_sentence(s)
                                            
                            except:
                                continue
                
                page.pubsub.send_all({'type': 'think_end'})
                
                if is_tts_enabled and not DEBUG_SKIP_TTS and not stop_event.is_set():
                    rem = sentence_buffer.flush()
                    if rem: backend.tts.queue_sentence(rem)
                
                messages.append({'role': 'assistant', 'content': full_response})

            else:
                result = backend.execute_function(func_name, params)
                page.pubsub.send_all({'type': 'simple_response', 'text': result})

                if is_tts_enabled:
                    import re
                    clean = re.sub(r'[^\w\s.,!?-]', '', result)
                    backend.tts.queue_sentence(clean)

        except Exception as e:
            page.pubsub.send_all({'type': 'error', 'text': str(e)})
        
        finally:
            page.pubsub.send_all({'type': 'done'})


    # --- Layout ---
    app_bar = ft.Row([
        ft.Text("Pocket AI", size=20, weight=ft.FontWeight.BOLD),
        ft.Container(expand=True),
        ft.IconButton(ft.Icons.CLEAN_HANDS_ROUNDED, tooltip="Clear Chat", on_click=clear_chat, icon_color=ft.Colors.GREY_500),
        ft.Text("Voice", size=12, color=ft.Colors.GREY_400),
        ft.Switch(value=True, on_change=toggle_tts, scale=0.8)
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

    input_bar = ft.Container(
        content=ft.Row([
            user_input,
            stop_button,
            send_button
        ]),
        padding=ft.Padding.only(top=10)
    )

    page.add(
        ft.Column([
            app_bar,
            status_text,
            ft.Divider(color=ft.Colors.GREY_800),
            chat_list,
            input_bar
        ], expand=True)
    )

    # --- Initial Preload ---
    def preload_background():
        status_text.value = "Warming up models..."
        page.update()
        backend.preload_models()
        if backend.tts.toggle(True):
            status_text.value = "Ready | TTS Active"
        else:
            status_text.value = "Ready | TTS Failed"
        page.update()

    threading.Thread(target=preload_background, daemon=True).start()

ft.app(target=main)
