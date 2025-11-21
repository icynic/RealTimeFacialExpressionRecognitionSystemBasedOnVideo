import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2 as cv
import mediapipe as mp
import numpy as np
import time
from classifier import ExpressionClassifier


class MainAppGui:
    def __init__(self, root, debug=False):
        self.root = root
        self.root.title("表情识别")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Configuration
        self.mp_model_path = "models/face_landmarker.task"
        self.classifier_model_path = "models/expression_classifier.tflite"
        self.categories_path = "blendshapes/categories.csv"

        # --- State Variables ---
        self.cap = None
        self.is_running = False
        self.update_id = None
        self.landmarker = None
        self.classifier = None
        self.app_start_time = 0
        self.latest_face_data = [] # Stores dicts: [{'box': (x,y,w,h), 'expression': 'text'}, ...]

        self.cap_width = 640
        self.cap_height = 480

        # Camera and settings variables
        self.available_cameras = []
        self.cap_device = 0 # Default to camera 0
        self.selected_camera_var = tk.StringVar()
        self.max_faces_var = tk.IntVar(value=5)
        self.current_landmarker_max_faces = None # Store the max_faces used for the current landmarker
        self.process_freq_var = tk.IntVar(value=1)
        self.frame_count = 0

        # GUI Elements
        self.video_label = None
        self.start_button = None
        self.stop_button = None
        self.expression_display_label = None
        self.placeholder_photo = None
        self.camera_selector = None
        self.max_faces_spinbox = None
        self.process_freq_spinbox = None

        self.setup_gui()
        self.load_classifier() # Load classifier first
        # Landmarker will be initialized on first start or when max_faces changes
        self.debug=debug
        self.last_cap_time=0

    def load_classifier(self):
        """Loads only the ExpressionClassifier."""
        try:
            if ExpressionClassifier:
                self.classifier = ExpressionClassifier(self.classifier_model_path, self.categories_path)
                print("ExpressionClassifier loaded successfully.")
                return True
            else:
                print("ExpressionClassifier not imported. Prediction functionality will be disabled.")
                messagebox.showwarning("分类器缺失",
                                       "无法导入表情分类器。视频将运行但没有表情预测功能。")
                return False
        except Exception as e:
            error_msg = f"Failed to load ExpressionClassifier: {e}"
            print(f"Error: {error_msg}")
            messagebox.showerror("分类器错误", error_msg)
            return False

    def initialize_landmarker(self):
        """Initializes or re-initializes the MediaPipe FaceLandmarker."""
        if self.landmarker:
            print("Closing existing FaceLandmarker...")
            self.landmarker.close()
            self.landmarker = None
            self.current_landmarker_max_faces = None # Reset stored value
            self.latest_face_data = [] # Clear face data on re-initialization

        current_max_faces_setting = self.max_faces_var.get()
        print(f"Initializing FaceLandmarker with max_faces={current_max_faces_setting}...")
        try:
            base_options = mp.tasks.BaseOptions(model_asset_path=self.mp_model_path)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                result_callback=self.result_callback_gui,
                output_face_blendshapes=True,
                num_faces=current_max_faces_setting, # Use variable here
            )
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self.current_landmarker_max_faces = current_max_faces_setting # Store the value used
            print("FaceLandmarker failed to initialize")
            return True
        except Exception as e:
            error_msg = f"没有成功初始化FaceLandmarker: {e}"
            print(f"Error: {error_msg}")
            messagebox.showerror("Landmarker 错误", error_msg)
            self.landmarker = None
            self.current_landmarker_max_faces = None
            return False

    def result_callback_gui(self, result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
        if not self.is_running: # If application is stopping, do nothing.
            return

        new_face_data = []

        img_h = output_image.height
        img_w = output_image.width

        if result.face_landmarks:
            for i, landmarks_list in enumerate(result.face_landmarks):
                if not landmarks_list:
                    continue

                # Calculate bounding box
                min_x = landmarks_list[0].x
                max_x = landmarks_list[0].x
                min_y = landmarks_list[0].y
                max_y = landmarks_list[0].y

                for landmark in landmarks_list:
                    min_x = min(min_x, landmark.x)
                    max_x = max(max_x, landmark.x)
                    min_y = min(min_y, landmark.y)
                    max_y = max(max_y, landmark.y)

                origin_x = int(min_x * img_w)
                origin_y = int(min_y * img_h)
                width = int((max_x - min_x) * img_w)
                height = int((max_y - min_y) * img_h)
                box = (origin_x, origin_y, width, height)

                current_expression = "N/A" # Default per-face expression
                if result.face_blendshapes and i < len(result.face_blendshapes):
                    if self.classifier:
                        face_blendshapes_data = result.face_blendshapes[i]
                        feature_vector = np.array(
                            [blendshape_category.score for blendshape_category in face_blendshapes_data]
                        )
                        
                        
                        if self.debug and self.last_cap_time:
                            extracting_time=time.time()-self.last_cap_time
                            print("Blendshapes extracting time(ms): ", extracting_time*1000)
                        
                        
                        
                        prediction = self.classifier.predict(feature_vector)
                        current_expression = str(prediction)
                        
                        if self.debug and self.last_cap_time:
                            print("classifying time: ", (time.time()-self.last_cap_time-extracting_time)*1000)
                        
                    else:
                        current_expression = "分类器未加载"
                
                new_face_data.append({'box': box, 'expression': current_expression})
        
        self.latest_face_data = new_face_data
        
        if self.debug and self.last_cap_time:
            # print("Delay since frame captured(ms): ", (time.time()-self.last_cap_time)*1000)
            pass

        # Generate display summary text
        display_summary_text: str
        if not new_face_data:
            if not result.face_landmarks: # No faces detected by landmarker
                display_summary_text = "状态: 未检测到人脸"
            else: # Faces detected by landmarker, but processing failed for all (e.g. no blendshapes)
                display_summary_text = "状态: 正在处理检测到的人脸..."
        else:
            expressions = [face['expression'] for face in new_face_data if 'expression' in face]
            if not expressions: # Should be rare if new_face_data is not empty and items have 'expression'
                display_summary_text = "状态: 正在处理检测到的人脸..."
            else:
                counts = {}
                for expr in expressions:
                    counts[expr] = counts.get(expr, 0) + 1
                
                expr_summary_parts = []
                for expr, count in counts.items():
                    if count > 1:
                        expr_summary_parts.append(f"{expr} (x{count})")
                    else:
                        expr_summary_parts.append(expr)
                display_summary_text = f"表情: {', '.join(expr_summary_parts)}"
        
        # Schedule GUI update only if still running and root window exists
        if self.is_running:
            try:
                if self.root.winfo_exists():
                    self.root.after(0, self.update_expression_display_gui, display_summary_text)
            except tk.TclError:
                # This can happen if the root window is being destroyed.
                # Silently ignore if we're in the process of closing.
                pass

    def update_expression_display_gui(self, display_text: str):
        # Check if still running and if the widget/root window still exist
        if not self.is_running:
            return
        
        try:
            if self.expression_display_label and \
               self.expression_display_label.winfo_exists() and \
               self.root and self.root.winfo_exists():
                self.expression_display_label.config(text=display_text)
        except tk.TclError:
            # This can happen if the widget or root window was destroyed
            # between scheduling and execution. Silently ignore during shutdown.
            pass

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        placeholder_img = Image.new('RGB', (self.cap_width, self.cap_height), color='black')
        self.placeholder_photo = ImageTk.PhotoImage(placeholder_img)

        self.video_label = ttk.Label(main_frame, image=self.placeholder_photo, anchor='center')
        self.video_label.image = self.placeholder_photo
        self.video_label.grid(row=0, column=0, columnspan=5, sticky=(tk.W, tk.E, tk.N, tk.S)) # Span more columns
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Control Frame
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=1, column=0, columnspan=5, pady=5, sticky=(tk.W, tk.E))

        self.start_button = ttk.Button(control_frame, text="开始", command=self.start_capture, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="停止", command=self.stop_capture, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Camera Selection
        camera_label = ttk.Label(control_frame, text="摄像头")
        camera_label.pack(side=tk.LEFT, padx=(10, 2))
        self.camera_selector = ttk.Combobox(
            control_frame, textvariable=self.selected_camera_var, state='readonly', width=15
        )
        self.camera_selector.bind('<<ComboboxSelected>>', self.on_camera_select)
        self.camera_selector.pack(side=tk.LEFT, padx=2)

        # Max Faces Spinbox
        max_faces_label = ttk.Label(control_frame, text="最大人脸数：")
        max_faces_label.pack(side=tk.LEFT, padx=(10, 2))
        self.max_faces_spinbox = ttk.Spinbox(
            control_frame, from_=1, to=1000, textvariable=self.max_faces_var, width=3, state='readonly'
        )
        self.max_faces_spinbox.pack(side=tk.LEFT, padx=2)

        # Processing Frequency Spinbox
        process_freq_label = ttk.Label(control_frame, text="处理频率（帧）：")
        process_freq_label.pack(side=tk.LEFT, padx=(10, 2))
        self.process_freq_spinbox = ttk.Spinbox(
            control_frame, from_=1, to=1000, textvariable=self.process_freq_var, width=3, state='readonly'
        )
        self.process_freq_spinbox.pack(side=tk.LEFT, padx=2)


        self.expression_display_label = ttk.Label(main_frame, text="状态: 初始化中...", font=("Helvetica", 12))
        self.expression_display_label.grid(row=2, column=0, columnspan=5, pady=5, sticky=(tk.W, tk.E))

        # Initial detect cameras
        self.detect_and_populate_cameras()


    def detect_and_populate_cameras(self):
        print("Detecting cameras...")
        self.available_cameras = []
        index = 0
        while True:
            # For Windows, cv.CAP_DSHOW can be more reliable. For others, default is fine.
            # Using a try-except block for platform-specific backend if needed.
            cap_test = cv.VideoCapture(index, cv.CAP_DSHOW if hasattr(cv, 'CAP_DSHOW') else index)
            if cap_test.isOpened():
                self.available_cameras.append(index)
                cap_test.release()
                print(f"Found camera: {index}")
                index += 1
                if index > 5: break # Limit search to save time
            else:
                cap_test.release()
                print(f"No camera at index: {index}. Search ended.")
                break
        
        camera_options = [f"摄像头 {i}" for i in self.available_cameras]
        if camera_options:
            self.camera_selector['values'] = camera_options
            if not self.selected_camera_var.get() or self.cap_device not in self.available_cameras:
                self.cap_device = self.available_cameras[0]
                self.selected_camera_var.set(camera_options[0])
            print(f"Default/selected camera: {self.cap_device}")
            self.start_button.config(state=tk.NORMAL)
            self.expression_display_label.config(text="状态: 就绪，请点击开始。")
        else:
            print("No cameras detected.")
            self.selected_camera_var.set("未找到摄像头")
            self.camera_selector['values'] = []
            self.start_button.config(state=tk.DISABLED)
            self.expression_display_label.config(text="状态: 错误 - 未检测到摄像头。")

    def on_camera_select(self, event=None):
        selected_option = self.selected_camera_var.get()
        try:
            self.cap_device = int(selected_option.split()[-1])
            print(f"Selected camera device: {self.cap_device}")
            if not self.is_running:
                 self.start_button.config(state=tk.NORMAL) # Enable start if not running
        except (IndexError, ValueError):
            print(f"Error parsing camera selection: {selected_option}")
            self.cap_device = -1 # Invalid selection
            self.start_button.config(state=tk.DISABLED)


    def start_capture(self):
        if self.is_running: return

        if self.cap_device == -1 or self.cap_device not in self.available_cameras:
             messagebox.showerror("摄像头错误", "未选择有效摄像头或摄像头不可用。")
             return

        # Always close existing landmarker (if any) and initialize a new one.
        if not self.initialize_landmarker():
            self.expression_display_label.config(text="状态: 错误 - Landmarker初始化失败。")
            # The initialize_landmarker method already shows an error messagebox on failure.
            return # Stop if landmarker failed


        self.cap = cv.VideoCapture(self.cap_device, cv.CAP_DSHOW if hasattr(cv, 'CAP_DSHOW') else self.cap_device)
        if not self.cap.isOpened():
            messagebox.showerror("摄像头错误", f"无法打开摄像头 {self.cap_device}。")
            self.cap = None
            return

        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

        self.app_start_time = time.time()
        self.is_running = True
        self.frame_count = 0
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.camera_selector.config(state='disabled')
        self.max_faces_spinbox.config(state='disabled')
        self.process_freq_spinbox.config(state='disabled')
        self.expression_display_label.config(text="状态: 运行中...")

        self.update_frame()

    def stop_capture(self):
        if not self.is_running: return
        self.is_running = False
        if self.update_id:
            self.root.after_cancel(self.update_id)
            self.update_id = None

        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.latest_face_data = [] # Clear face data when stopping

        self.video_label.config(image=self.placeholder_photo)
        self.video_label.image = self.placeholder_photo

        self.start_button.config(state=tk.NORMAL if self.available_cameras else tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.camera_selector.config(state='readonly')
        self.max_faces_spinbox.config(state='readonly')
        self.process_freq_spinbox.config(state='readonly')
        self.expression_display_label.config(text="状态: 已停止，准备开始。")

    def update_frame(self):
        if not self.is_running or not self.cap or not self.cap.isOpened():
            if self.is_running: self.stop_capture()
            return

        success, frame = self.cap.read()
        if not success:
            print("Camera frame unavailable. Stopping.")
            self.expression_display_label.config(text="状态: 错误 - 摄像头已断开？")
            self.stop_capture()
            return
        
        
        if self.debug:
            self.last_cap_time=time.time()
        
        
        self.frame_count += 1
        process_this_frame = (self.frame_count % self.process_freq_var.get() == 0)

        frame_flipped = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame_flipped, cv.COLOR_BGR2RGB)

        if process_this_frame and self.landmarker:
            

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            current_timestamp_ms = int((time.time() - self.app_start_time) * 1000)
            try:
                self.landmarker.detect_async(mp_image, current_timestamp_ms)
            except Exception as e:
                print(f"Error in detect_async: {e}")
                # Could potentially stop or show an error on GUI if this persists

        # Draw rectangles and text on the frame for each detected face
        faces_to_draw = list(self.latest_face_data) # Make a copy
        for face_data in faces_to_draw:
            x, y, w, h = face_data['box']
            expression = face_data['expression']

            # Draw rectangle
            start_point = (int(x), int(y))
            end_point = (int(x + w), int(y + h))
            cv.rectangle(frame_rgb, start_point, end_point, (0, 255, 0), 2) # Green rectangle

            # Prepare text properties
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (255, 255, 255) # White
            line_type = 1
            text_x = int(x)
            text_y = int(y) - 10 # Position text slightly above the rectangle
            if text_y < 10: # Ensure text is not drawn off-screen (top)
                text_y = int(y) + int(h) + 20


            # Add a background to the text for better visibility
            (text_width, text_height), baseline = cv.getTextSize(expression, font, font_scale, line_type)
            cv.rectangle(frame_rgb, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0,0,0), -1) # Black background

            cv.putText(frame_rgb, expression, (text_x, text_y), font, font_scale, font_color, line_type)


        # Display the current frame
        img_pil = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        if self.is_running:
            self.update_id = self.root.after(15, self.update_frame) # UI refresh rate

    def on_closing(self):
        print("Closing application...")
        self.stop_capture()
        if self.landmarker:
            self.landmarker.close()
            print("FaceLandmarker closed.")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainAppGui(root)
    # app = MainAppGui(root, debug=True)
    root.mainloop()