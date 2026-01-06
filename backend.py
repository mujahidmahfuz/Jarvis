import requests
import json
import sys
import os
import re
import wave
import threading
import queue
import time
import sounddevice as sd
import numpy as np
from pathlib import Path

# ANSI Escape Codes for coloring output (kept for console logging)
GRAY = "\033[90m"
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

# --- Model Configuration ---
RESPONDER_MODEL = "qwen3:1.7b"       # Conversational responses
OLLAMA_URL = "http://localhost:11434/api"
LOCAL_ROUTER_PATH = "./merged_model"
MAX_HISTORY = 20

# Persistent Session for faster HTTP
http_session = requests.Session()

try:
    from function_router import FunctionGemmaRouter
except ImportError:
    print(f"{GRAY}[System] FunctionGemmaRouter not found.{RESET}")
    sys.exit(1)

# Global Router Instance
router = None

# Keywords that trigger the Router (otherwise we default to chat)
ROUTER_KEYWORDS = [
    # Tools
    "turn", "light", "dim", "switch",   # Lights
    "search", "google", "find", "look", # Search
    "timer", "alarm", "clock",          # Timers
    "calendar", "schedule", "appoint", "meet", "event", # Calendar
    
    # Complexity / Thinking Triggers (from Training Data)
    "explain", "how", "why", "cause", "difference", "compare", "meaning", # Reasoning
    "solve", "calculate", "equation", "math", "+", "*", "divide", "minus", # Math
    "write", "poem", "haiku", "riddle", "story", "script", "code", # Creative/Coding
    "if", "when" # Conditionals
]

def should_bypass_router(text):
    """Return True if text definitely doesn't need routing."""
    text = text.lower()
    return not any(k in text for k in ROUTER_KEYWORDS)

# --- Function Definitions (Official JSON Schema) ---
FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "control_light",
            "description": "Controls smart lights - turn on, off, or dim lights in a room",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "The action to perform: on, off, or dim"},
                    "room": {"type": "string", "description": "The room name where the light is located"}
                },
                "required": ["action", "room"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web for information using Google",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Sets a countdown timer for a specified duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {"type": "string", "description": "Time duration like 5 minutes or 1 hour"},
                    "label": {"type": "string", "description": "Optional timer name or label"}
                },
                "required": ["duration"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Creates a new calendar event or appointment",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The event title"},
                    "date": {"type": "string", "description": "The date of the event"},
                    "time": {"type": "string", "description": "The time of the event"},
                    "description": {"type": "string", "description": "Optional event details"}
                },
                "required": ["title", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_calendar",
            "description": "Reads and retrieves calendar events for a date or time range",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "The date or date range to check"},
                    "filter": {"type": "string", "description": "Optional filter like meetings or appointments"}
                },
                "required": ["date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "passthrough",
            "description": "DEFAULT FUNCTION - Use this whenever no other function is clearly needed. This is the fallback for: greetings (hello, hi, good morning), chitchat (how are you, what's your name), general knowledge questions, explanations, conversations, and ANY query that does NOT explicitly require controlling lights, setting timers, searching the web, or managing calendar events. When in doubt, use passthrough.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {"type": "boolean", "description": "Set to true for complex reasoning/math/logic, false for simple greetings and chitchat."}
                },
                "required": ["thinking"]
            }
        }
    }
]

def route_query(user_input):
    """Route user query using local FunctionGemmaRouter."""
    global router
    if not router:
        return "passthrough", {"thinking": False}

    try:
        # Route using the fine-tuned model (thinking vs nonthinking)
        decision, elapsed = router.route_with_timing(user_input)
        
        # Map to passthrough params
        if decision == "thinking":
            return "passthrough", {"thinking": True, "router_latency": elapsed}
        else: # nonthinking
            return "passthrough", {"thinking": False, "router_latency": elapsed}
            
    except Exception as e:
        print(f"{GRAY}[Router Error: {e}]{RESET}")
        return "passthrough", {"thinking": False, "router_latency": 0.0}

# --- Function Execution Stubs ---
def execute_function(name, params):
    """Execute function and return response string."""
    if name == "control_light":
        action = params.get("action", "toggle")
        room = params.get("room", "room")
        if action == "on":
            return f"💡 Turned on the {room} lights."
        elif action == "off":
            return f"💡 Turned off the {room} lights."
        elif action == "dim":
            return f"💡 Dimmed the {room} lights."
        else:
            return f"💡 {action.capitalize()} the {room} lights."
    
    elif name == "web_search":
        query = params.get("query", "")
        return f"🔍 Searching the web for: {query}"
    
    elif name == "set_timer":
        duration = params.get("duration", "")
        label = params.get("label", "Timer")
        return f"⏱️ Timer set for {duration}" + (f" ({label})" if label else "")
    
    elif name == "create_calendar_event":
        title = params.get("title", "Event")
        date = params.get("date", "")
        time = params.get("time", "")
        return f"📅 Created event: {title} on {date}" + (f" at {time}" if time else "")
    
    elif name == "read_calendar":
        date = params.get("date", "today")
        return f"📆 Checking calendar for {date}..."
    
    else:
        return f"Unknown function: {name}"


# --- Piper TTS Integration ---
class PiperTTS:
    """Lightweight Piper TTS wrapper with streaming sentence support."""
    
    VOICE_MODEL = "en_GB-alba-medium"
    MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx"
    CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json"
    
    def __init__(self):
        self.enabled = False
        self.voice = None
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self.interrupt_event = threading.Event()
        self.models_dir = Path.home() / ".local" / "share" / "piper" / "voices"
        
        try:
            from piper import PiperVoice
            self.PiperVoice = PiperVoice
            self.available = True
        except ImportError:
            self.available = False
            print(f"{GRAY}[TTS] piper-tts not installed. Run: pip install piper-tts{RESET}")
    
    def download_model(self):
        """Download voice model if not present."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.models_dir / f"{self.VOICE_MODEL}.onnx"
        config_path = self.models_dir / f"{self.VOICE_MODEL}.onnx.json"
        
        if not model_path.exists():
            print(f"{CYAN}[TTS] Downloading voice model ({self.VOICE_MODEL})...{RESET}")
            r = http_session.get(self.MODEL_URL, stream=True)
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            r = http_session.get(self.CONFIG_URL)
            r.raise_for_status()
            with open(config_path, 'wb') as f:
                f.write(r.content)
            print(f"{CYAN}[TTS] Model downloaded!{RESET}")
        
        return str(model_path), str(config_path)
    
    def initialize(self):
        """Load the voice model."""
        if not self.available:
            return False
        
        try:
            model_path, config_path = self.download_model()
            self.voice = self.PiperVoice.load(model_path, config_path)
            self.running = True
            self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.worker_thread.start()
            return True
        except Exception as e:
            print(f"{GRAY}[TTS] Failed to initialize: {e}{RESET}")
            return False
    
    def _speech_worker(self):
        """Background thread that plays queued sentences."""
        while self.running:
            try:
                # Check for interrupt before getting next sentence
                if self.interrupt_event.is_set():
                    self.interrupt_event.clear()
                
                text = self.speech_queue.get(timeout=0.5)
                if text is None:
                    break
                
                # Check again
                if self.interrupt_event.is_set():
                    self.speech_queue.task_done()
                    continue

                self._speak_text(text)
                self.speech_queue.task_done()
            except queue.Empty:
                continue
    
    def _speak_text(self, text):
        """Synthesize and play text using sounddevice streaming."""
        if not self.voice or not text.strip():
            return
        
        try:
            sample_rate = self.voice.config.sample_rate
            
            # Stream audio directly to output device
            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype='int16', latency='low') as stream:
                self.current_stream = stream
                for audio_chunk in self.voice.synthesize(text):
                    if self.interrupt_event.is_set():
                        stream.abort() # Instant stop
                        break 
                    data = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
                    stream.write(data)
                self.current_stream = None
                    
        except Exception as e:
            print(f"{GRAY}[TTS Error]: {e}{RESET}")
    
    def queue_sentence(self, sentence):
        """Add a sentence to the speech queue."""
        if self.enabled and self.voice and sentence.strip():
            self.speech_queue.put(sentence)
            
    def stop(self):
        """Interrupt current speech and clear queue."""
        self.interrupt_event.set()
        # Clear queue safely
        with self.speech_queue.mutex:
            self.speech_queue.queue.clear()
        # Abort active stream
        if hasattr(self, 'current_stream') and self.current_stream:
            try:
                self.current_stream.abort()
            except:
                pass
            
    def wait_for_completion(self):
        """Wait for all queued speech to finish."""
        if self.enabled:
            self.speech_queue.join()
    
    def toggle(self, enable):
        """Enable/disable TTS."""
        if enable and not self.voice:
            if self.initialize():
                self.enabled = True
                return True
            return False
        self.enabled = enable
        return True
    
    def shutdown(self):
        """Clean up resources."""
        self.running = False
        self.stop()
        self.speech_queue.put(None)


class SentenceBuffer:
    """Buffers streaming text and extracts complete sentences."""
    
    SENTENCE_ENDINGS = re.compile(r'([.!?])\s+|([.!?])$')
    
    def __init__(self):
        self.buffer = ""
    
    def add(self, text):
        """Add text chunk and return any complete sentences."""
        self.buffer += text
        sentences = []
        
        while True:
            match = self.SENTENCE_ENDINGS.search(self.buffer)
            if match:
                end_pos = match.end()
                sentence = self.buffer[:end_pos].strip()
                if sentence:
                    sentences.append(sentence)
                self.buffer = self.buffer[end_pos:]
            else:
                break
        
        return sentences
    
    def flush(self):
        """Return any remaining text as a final sentence."""
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None


# Global TTS instance
tts = PiperTTS()


# --- Model Preloading ---
def preload_models():
    """Client-side preload to ensure models are in memory before user interaction. Parallelized."""
    print(f"{GRAY}[System] Preloading models...{RESET}")
    
    threads = []

    def load_router():
        global router
        try:
            router = FunctionGemmaRouter(model_path=LOCAL_ROUTER_PATH, compile_model=False)
            # Warm up
            router.route("Hello")
        except Exception as e:
            print(f"{GRAY}[Router] Failed to load local model: {e}{RESET}")

    def load_responder():
        try:
            http_session.post(f"{OLLAMA_URL}/chat", json={
                "model": RESPONDER_MODEL, 
                "messages": [], 
                "keep_alive": "5m"
            }, timeout=1)
        except:
            pass

    def load_voice():
        print(f"{GRAY}[System] Loading voice model...{RESET}")
        tts.initialize()

    # Create threads
    threads.append(threading.Thread(target=load_router))
    threads.append(threading.Thread(target=load_responder))
    threads.append(threading.Thread(target=load_voice))

    # Start all
    for t in threads:
        t.start()
    
    # Wait for all
    for t in threads:
        t.join()

    print(f"{GRAY}[System] Models warm and ready.{RESET}")
