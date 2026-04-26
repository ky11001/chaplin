import cv2
import time
import numpy as np
from collections import deque
from ollama import AsyncClient
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from pynput import keyboard
import asyncio


class ChaplinOutput(BaseModel):
    list_of_changes: str
    corrected_text: str


class Chaplin:
    def __init__(self):
        self.vsr_model = None

        # flag to toggle recording
        self.recording = False

        # thread stuff
        self.executor = ThreadPoolExecutor(max_workers=1)

        # video params
        self.res_factor = 3
        self.fps = 16
        self.frame_interval = 1 / self.fps

        # streaming params: while recording, flush a chunk every `chunk_seconds`
        # and seed the next chunk with `overlap_frames` carried over so words
        # straddling chunk boundaries still have context.
        self.chunk_seconds = 2.0
        self.chunk_frames = int(self.fps * self.chunk_seconds)
        self.overlap_frames = 5

        # setup keyboard controller for typing
        self.kbd_controller = keyboard.Controller()

        # setup async ollama client
        self.ollama_client = AsyncClient()

        # setup asyncio event loop in background thread
        self.loop = asyncio.new_event_loop()
        self.async_thread = ThreadPoolExecutor(max_workers=1)
        self.async_thread.submit(self._run_event_loop)

        # sequence tracking to ensure outputs are typed in order
        self.next_sequence_to_type = 0
        self.current_sequence = 0  # counter for assigning sequence numbers
        self.typing_lock = None  # will be created in async loop
        self._init_async_resources()

        # setup global hotkey for toggling recording with option/alt key
        self.hotkey = keyboard.GlobalHotKeys({
            '<alt>': self.toggle_recording
        })
        self.hotkey.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _init_async_resources(self):
        """Initialize async resources in the async loop"""
        future = asyncio.run_coroutine_threadsafe(
            self._create_async_lock(), self.loop)
        future.result()  # wait for it to complete

    async def _create_async_lock(self):
        """Create asyncio.Lock and Condition in the event loop's context"""
        self.typing_lock = asyncio.Lock()
        self.typing_condition = asyncio.Condition(self.typing_lock)

    def toggle_recording(self):
        # toggle recording when alt/option key is pressed
        self.recording = not self.recording

    async def correct_output_async(self, output, sequence_num):
        # perform inference on the raw output to get back a "correct" version
        response = await self.ollama_client.chat(
            model='qwen3:4b',
            messages=[
                {
                    'role': 'system',
                    'content': f"You are an assistant that helps make corrections to the output of a lipreading model. The text you will receive was transcribed using a video-to-text system that attempts to lipread the subject speaking in the video, so the text will likely be imperfect. The input text will also be in all-caps, although your respose should be capitalized correctly and should NOT be in all-caps.\n\nIf something seems unusual, assume it was mistranscribed. Do your best to infer the words actually spoken, and make changes to the mistranscriptions in your response. Do not add more words or content, just change the ones that seem to be out of place (and, therefore, mistranscribed). Do not change even the wording of sentences, just individual words that look nonsensical in the context of all of the other words in the sentence.\n\nAlso, add correct punctuation to the entire text. ALWAYS end each sentence with the appropriate sentence ending: '.', '?', or '!'. \n\nReturn the corrected text in the format of 'list_of_changes' and 'corrected_text'."
                },
                {
                    'role': 'user',
                    'content': f"Transcription:\n\n{output}"
                }
            ],
            format=ChaplinOutput.model_json_schema()
        )

        # get only the corrected text
        chat_output = ChaplinOutput.model_validate_json(
            response['message']['content'])

        # if last character isn't a sentence ending (happens sometimes), add a period
        chat_output.corrected_text = chat_output.corrected_text.strip()
        if chat_output.corrected_text[-1] not in ['.', '?', '!']:
            chat_output.corrected_text += '.'

        # add space at the end
        chat_output.corrected_text += ' '

        # wait until it's this task's turn to type
        async with self.typing_condition:
            while self.next_sequence_to_type != sequence_num:
                await self.typing_condition.wait()

            # this task's turn to type the corrected text
            self.kbd_controller.type(chat_output.corrected_text)

            # increment sequence and notify next task
            self.next_sequence_to_type += 1
            self.typing_condition.notify_all()

        return chat_output.corrected_text

    def _detect_landmarks(self, rgb_frame):
        """Run mediapipe on a single RGB frame; fall back to short-range."""
        det = self.vsr_model.landmarks_detector
        lm = det.detect([rgb_frame], det.full_range_detector)[0]
        if lm is None:
            lm = det.detect([rgb_frame], det.short_range_detector)[0]
        return lm

    def perform_inference(self, frames, landmarks):
        # bail if mediapipe never locked on during this chunk — the downstream
        # interpolator asserts at least one valid frame.
        if not any(lm is not None for lm in landmarks):
            print("\n\033[48;5;208m\033[97m\033[1m SKIPPED \033[0m: no face detected in chunk\n")
            return {"output": ""}

        video = np.stack(frames, axis=0)
        output = self.vsr_model.infer_arrays(video, landmarks)

        print(f"\n\033[48;5;21m\033[97m\033[1m RAW OUTPUT \033[0m: {output}\n")

        text = (output or "").strip()
        if text:
            self.kbd_controller.type(text + " ")

        return {"output": output}

    def start_webcam(self):
        # init webcam
        cap = cv2.VideoCapture(0)

        # set webcam resolution, and get frame dimensions
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 // self.res_factor)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 // self.res_factor)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        last_frame_time = time.time()

        futures = []
        # in-memory buffers; no disk round-trip.
        chunk_frames = []
        chunk_landmarks = []
        # rolling tail used to seed the next chunk with overlap so words
        # crossing chunk boundaries survive.
        overlap_frames = deque(maxlen=self.overlap_frames)
        overlap_landmarks = deque(maxlen=self.overlap_frames)

        def _flush_chunk():
            """Submit the current chunk if it has at least 2s of content."""
            if len(chunk_frames) >= self.fps * 2:
                futures.append(self.executor.submit(
                    self.perform_inference, list(chunk_frames), list(chunk_landmarks)))

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            current_time = time.time()
            if current_time - last_frame_time < self.frame_interval:
                continue

            ret, frame = cap.read()
            if not ret:
                continue

            # capture is BGR; mediapipe + the model preprocessing both expect RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.recording:
                # seed a fresh chunk with the previous chunk's overlap tail.
                if not chunk_frames and overlap_frames:
                    chunk_frames.extend(overlap_frames)
                    chunk_landmarks.extend(overlap_landmarks)

                lm = self._detect_landmarks(rgb_frame)
                chunk_frames.append(rgb_frame)
                chunk_landmarks.append(lm)
                overlap_frames.append(rgb_frame)
                overlap_landmarks.append(lm)

                last_frame_time = current_time

                # mid-recording flush so transcription streams.
                if len(chunk_frames) >= self.chunk_frames:
                    _flush_chunk()
                    chunk_frames.clear()
                    chunk_landmarks.clear()

                # display: BGR copy with record indicator.
                display = frame.copy()
                cv2.circle(display, (frame_width - 20, 20), 10, (0, 0, 255), -1)
            elif chunk_frames:
                # recording just stopped — flush the tail and reset.
                _flush_chunk()
                chunk_frames.clear()
                chunk_landmarks.clear()
                overlap_frames.clear()
                overlap_landmarks.clear()
                display = frame
            else:
                display = frame

            cv2.imshow('Chaplin', cv2.flip(display, 1))

            # ensure outputs are consumed in capture order.
            for fut in futures:
                if fut.done():
                    fut.result()
                    futures.remove(fut)
                else:
                    break

        # release everything
        cap.release()
        cv2.destroyAllWindows()

        # stop global hotkey listener
        self.hotkey.stop()

        # stop async event loop
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.async_thread.shutdown(wait=True)

        # shutdown executor
        self.executor.shutdown(wait=True)
