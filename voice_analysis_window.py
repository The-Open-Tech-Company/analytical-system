"""
Модуль интерфейса для анализа голоса
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import threading
import sounddevice as sd
import soundfile as sf
from voice_analyzer import VoiceAnalyzer
from voice_visualizer import VoiceVisualizer


class VoiceAnalysisWindow:
    """Класс для создания окна анализа голоса"""
    
    def __init__(self, parent_window, main_app):
        self.parent_window = parent_window
        self.main_app = main_app
        self.parent_window.title("Анализ голоса")
        self.parent_window.geometry("1200x800")
        self.parent_window.configure(bg='#f0f0f0')
        
        # Переменные
        self.audio_path = None
        self.audio_data = None
        self.sample_rate = None
        self.analysis_result = None
        self.is_recording = False
        self.recording_thread = None
        self.recording_frames = []
        self.recording_start_time = None
        self.recording_timer_id = None
        
        # Инициализация компонентов
        try:
            self.analyzer = VoiceAnalyzer()
            self.visualizer = VoiceVisualizer()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка инициализации системы: {e}")
            return
        
        self.create_widgets()
    
    def create_widgets(self):
        """Создает виджеты интерфейса"""
        
        # Заголовок
        title_label = tk.Label(
            self.parent_window,
            text="Анализ голоса",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        title_label.pack(pady=10)
        
        # Основной контейнер
        main_frame = tk.Frame(self.parent_window, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - загрузка и управление
        left_panel = tk.Frame(main_frame, bg='#f0f0f0', width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5)
        
        # Панель загрузки
        load_frame = tk.LabelFrame(
            left_panel,
            text="Загрузка аудио",
            font=("Arial", 11, "bold"),
            bg='#f0f0f0',
            fg='#333',
            padx=10,
            pady=10
        )
        load_frame.pack(fill=tk.X, pady=5)
        
        load_btn = tk.Button(
            load_frame,
            text="📁 Загрузить файл",
            font=("Arial", 12, "bold"),
            bg='#2196F3',
            fg='white',
            padx=20,
            pady=10,
            command=self.load_audio_file,
            cursor='hand2'
        )
        load_btn.pack(pady=10)
        
        # Кнопка записи
        self.record_btn = tk.Button(
            load_frame,
            text="🎤 Начать запись",
            font=("Arial", 12, "bold"),
            bg='#F44336',
            fg='white',
            padx=20,
            pady=10,
            command=self.toggle_recording,
            cursor='hand2'
        )
        self.record_btn.pack(pady=10)
        
        # Индикатор записи с таймером
        self.recording_label = tk.Label(
            load_frame,
            text="",
            font=("Arial", 14, "bold"),
            bg='#f0f0f0',
            fg='red'
        )
        self.recording_label.pack(pady=5)
        
        # Прогресс-бар записи (визуальный индикатор)
        self.recording_progress = ttk.Progressbar(
            load_frame,
            mode='indeterminate',
            length=200
        )
        self.recording_progress.pack(pady=5)
        
        # Кнопка анализа
        analyze_btn = tk.Button(
            load_frame,
            text="🔍 Анализ",
            font=("Arial", 12, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            command=self.analyze_voice,
            cursor='hand2'
        )
        analyze_btn.pack(pady=10)
        
        # Информация о файле
        info_frame = tk.LabelFrame(
            left_panel,
            text="Информация о файле",
            font=("Arial", 11, "bold"),
            bg='#f0f0f0',
            fg='#333',
            padx=10,
            pady=10
        )
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.info_text = tk.Text(
            info_frame,
            height=10,
            font=("Arial", 9),
            bg='white',
            wrap=tk.WORD
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Правая панель - результаты
        right_panel = tk.Frame(main_frame, bg='#f0f0f0', width=700)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Область для результатов
        results_frame = tk.LabelFrame(
            right_panel,
            text="Результаты анализа",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#333',
            padx=10,
            pady=10
        )
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Canvas для прокрутки результатов
        canvas = tk.Canvas(results_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=canvas.yview)
        self.results_scrollable_frame = tk.Frame(canvas, bg='white')
        
        self.results_scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.results_scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.results_canvas = canvas
        self.results_frame = self.results_scrollable_frame
        
        # Кнопка визуализации
        viz_btn = tk.Button(
            right_panel,
            text="📊 Посмотреть визуализацию",
            font=("Arial", 12, "bold"),
            bg='#FF9800',
            fg='white',
            padx=20,
            pady=10,
            command=self.show_visualization,
            cursor='hand2'
        )
        viz_btn.pack(pady=10)
    
    def load_audio_file(self):
        """Загружает аудиофайл"""
        file_path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[
                ("Аудио файлы", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("WAV", "*.wav"),
                ("MP3", "*.mp3"),
                ("Все файлы", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Ошибка", f"Файл не найден:\n{file_path}")
            return
        
        try:
            self.audio_path = file_path
            self.audio_data, self.sample_rate = self.analyzer.load_audio(file_path)
            self.analysis_result = None
            
            # Обновляем информацию о файле
            duration = len(self.audio_data) / self.sample_rate
            info = f"Файл: {os.path.basename(file_path)}\n"
            info += f"Длительность: {duration:.2f} сек\n"
            info += f"Частота дискретизации: {self.sample_rate} Hz\n"
            info += f"Размер данных: {len(self.audio_data)} сэмплов"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
            
            self.clear_results()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить аудиофайл:\n{e}")
    
    def toggle_recording(self):
        """Переключает режим записи"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Начинает запись"""
        self.is_recording = True
        self.recording_frames = []
        self.recording_start_time = None
        self.record_btn.config(text="⏹ Остановить запись", bg='#4CAF50')
        self.recording_label.config(text="Запись: 00:00", fg='red')
        self.recording_progress.start(10)  # Анимация прогресс-бара
        
        # Запускаем запись в отдельном потоке
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Запускаем таймер обновления
        self._update_recording_timer()
    
    def stop_recording(self):
        """Останавливает запись"""
        self.is_recording = False
        self.recording_progress.stop()
        
        # Останавливаем таймер
        if self.recording_timer_id:
            self.parent_window.after_cancel(self.recording_timer_id)
            self.recording_timer_id = None
        
        self.record_btn.config(text="🎤 Начать запись", bg='#F44336')
        
        # Ждем завершения потока записи
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        # Объединяем все записанные фреймы
        if self.recording_frames:
            import time
            elapsed_time = time.time() - self.recording_start_time if self.recording_start_time else 0
            
            # Объединяем фреймы, если еще не объединены
            if self.audio_data is None and self.recording_frames:
                try:
                    self.audio_data = np.concatenate(self.recording_frames)
                    self.audio_path = None  # Это запись, не файл
                except Exception as e:
                    print(f"Ошибка объединения фреймов: {e}")
            
            self.recording_label.config(text=f"Запись завершена ({elapsed_time:.1f} сек)", fg='green')
            
            # Обновляем информацию - используем реальную длительность записи
            if self.audio_data is not None and self.sample_rate is not None:
                duration = len(self.audio_data) / self.sample_rate
            else:
                duration = elapsed_time
            
            info = f"Запись\n"
            info += f"Длительность: {duration:.2f} сек\n"
            info += f"Частота дискретизации: {self.sample_rate if self.sample_rate else 44100} Hz\n"
            if self.audio_data is not None:
                info += f"Размер данных: {len(self.audio_data)} сэмплов"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
            self.clear_results()
        else:
            self.recording_label.config(text="Запись не началась", fg='orange')
    
    def _update_recording_timer(self):
        """Обновляет таймер записи"""
        if not self.is_recording:
            return
        
        import time
        if self.recording_start_time:
            elapsed = time.time() - self.recording_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.recording_label.config(text=f"Запись: {minutes:02d}:{seconds:02d}")
        else:
            self.recording_label.config(text="Запись: 00:00")
        
        # Планируем следующее обновление через 1 секунду
        self.recording_timer_id = self.parent_window.after(1000, self._update_recording_timer)
    
    def _record_audio(self):
        """Записывает аудио (неограниченная запись)"""
        try:
            import time
            sample_rate = 44100
            chunk_size = 1024  # Размер чанка для записи
            
            self.recording_start_time = time.time()
            self.sample_rate = sample_rate
            
            # Открываем поток записи
            with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=chunk_size) as stream:
                while self.is_recording:
                    chunk, overflowed = stream.read(chunk_size)
                    if overflowed:
                        print("Предупреждение: переполнение буфера")
                    self.recording_frames.append(chunk.flatten())
            
            # Объединяем все фреймы
            if self.recording_frames:
                self.audio_data = np.concatenate(self.recording_frames)
                self.audio_path = None  # Это запись, не файл
            else:
                self.audio_data = None
            
        except Exception as e:
            self.parent_window.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка записи: {e}"))
            self.is_recording = False
    
    def analyze_voice(self):
        """Выполняет анализ голоса"""
        if self.audio_data is None:
            messagebox.showwarning(
                "Предупреждение",
                "Пожалуйста, сначала загрузите или запишите аудио."
            )
            return
        
        self.clear_results()
        
        loading_label = tk.Label(
            self.results_frame,
            text="Обработка аудио...",
            font=("Arial", 12),
            bg='white'
        )
        loading_label.pack(pady=20)
        self.parent_window.update()
        
        try:
            # Сохраняем временный файл, если это запись
            temp_file = None
            if self.audio_path is None:
                temp_file = "temp_recording.wav"
                sf.write(temp_file, self.audio_data, self.sample_rate)
                file_to_analyze = temp_file
            else:
                file_to_analyze = self.audio_path
            
            loading_label.config(text="Анализ голоса...")
            self.parent_window.update()
            
            self.analysis_result = self.analyzer.analyze_voice(file_to_analyze)
            
            # Удаляем временный файл
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            loading_label.destroy()
            self.display_results()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при анализе: {e}")
            if 'loading_label' in locals():
                loading_label.destroy()
    
    def clear_results(self):
        """Очищает область результатов"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
    
    def display_results(self):
        """Отображает результаты анализа"""
        if not self.analysis_result:
            return
        
        title_label = tk.Label(
            self.results_frame,
            text="Результаты анализа",
            font=("Arial", 14, "bold"),
            bg='white',
            fg='#333'
        )
        title_label.pack(pady=10)
        
        # Информация о поле
        gender = self.analysis_result.get('gender', 'Не определен')
        gender_conf = self.analysis_result.get('gender_confidence', 0.0)
        
        gender_frame = tk.Frame(self.results_frame, bg='white')
        gender_frame.pack(fill=tk.X, padx=10, pady=5)
        
        gender_label = tk.Label(
            gender_frame,
            text=f"Пол: {gender} (уверенность: {gender_conf*100:.0f}%)",
            font=("Arial", 12, "bold"),
            bg='white',
            fg='#333',
            anchor='w'
        )
        gender_label.pack(fill=tk.X)
        
        # Информация об акценте
        accent = self.analysis_result.get('accent', 'Не определен')
        accent_conf = self.analysis_result.get('accent_confidence', 0.0)
        
        accent_frame = tk.Frame(self.results_frame, bg='white')
        accent_frame.pack(fill=tk.X, padx=10, pady=5)
        
        accent_label = tk.Label(
            accent_frame,
            text=f"Акцент: {accent} (уверенность: {accent_conf*100:.0f}%)",
            font=("Arial", 12, "bold"),
            bg='white',
            fg='#333',
            anchor='w'
        )
        accent_label.pack(fill=tk.X)
        
        # Информация об эмоциях
        emotion = self.analysis_result.get('emotion', 'Не определена')
        emotion_conf = self.analysis_result.get('emotion_confidence', 0.0)
        
        emotion_frame = tk.Frame(self.results_frame, bg='white')
        emotion_frame.pack(fill=tk.X, padx=10, pady=5)
        
        emotion_label = tk.Label(
            emotion_frame,
            text=f"Эмоция: {emotion} (уверенность: {emotion_conf*100:.0f}%)",
            font=("Arial", 12, "bold"),
            bg='white',
            fg='#333',
            anchor='w'
        )
        emotion_label.pack(fill=tk.X)
        
        # Информация о языке
        language = self.analysis_result.get('language', 'Не определен')
        language_conf = self.analysis_result.get('language_confidence', 0.0)
        
        language_frame = tk.Frame(self.results_frame, bg='white')
        language_frame.pack(fill=tk.X, padx=10, pady=5)
        
        language_label = tk.Label(
            language_frame,
            text=f"Язык: {language} (уверенность: {language_conf*100:.0f}%)",
            font=("Arial", 12, "bold"),
            bg='white',
            fg='#333',
            anchor='w'
        )
        language_label.pack(fill=tk.X)
        
        # Разделитель
        separator = tk.Frame(self.results_frame, height=2, bg='#ccc')
        separator.pack(fill=tk.X, padx=10, pady=10)
        
        # Детальные характеристики
        features = self.analysis_result.get('features', {})
        
        features_title = tk.Label(
            self.results_frame,
            text="Детальные характеристики",
            font=("Arial", 12, "bold"),
            bg='white',
            fg='#333'
        )
        features_title.pack(pady=5)
        
        # Основная частота
        pitch_mean = features.get('pitch_mean', 0)
        if pitch_mean > 0:
            pitch_frame = tk.Frame(self.results_frame, bg='white')
            pitch_frame.pack(fill=tk.X, padx=10, pady=2)
            
            pitch_label = tk.Label(
                pitch_frame,
                text=f"Основная частота (pitch): {pitch_mean:.1f} Hz",
                font=("Arial", 10),
                bg='white',
                anchor='w'
            )
            pitch_label.pack(fill=tk.X)
        
        # Длительность
        duration = features.get('duration', 0)
        if duration > 0:
            duration_frame = tk.Frame(self.results_frame, bg='white')
            duration_frame.pack(fill=tk.X, padx=10, pady=2)
            
            duration_label = tk.Label(
                duration_frame,
                text=f"Длительность: {duration:.2f} сек",
                font=("Arial", 10),
                bg='white',
                anchor='w'
            )
            duration_label.pack(fill=tk.X)
        
        # Обновляем прокрутку
        self.results_canvas.update_idletasks()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        self.results_canvas.yview_moveto(0)
    
    def show_visualization(self):
        """Открывает окно с визуализациями"""
        if not self.analysis_result or self.audio_data is None:
            messagebox.showwarning("Предупреждение", "Сначала выполните анализ голоса.")
            return
        
        # Создаем новое окно для визуализаций
        viz_window = tk.Toplevel(self.parent_window)
        viz_window.transient(self.parent_window)
        viz_window.title("Визуализация анализа голоса")
        viz_window.geometry("1400x1000")
        viz_window.configure(bg='#f0f0f0')
        
        try:
            features = self.analysis_result.get('features', {})
            
            # Создаем комплексную визуализацию
            vis_img = self.visualizer.create_comprehensive_visualization(
                self.audio_data,
                self.sample_rate,
                features
            )
            
            # Конвертируем в PIL Image
            vis_pil = Image.fromarray(vis_img)
            vis_tk = ImageTk.PhotoImage(vis_pil)
            
            # Отображаем
            label = tk.Label(viz_window, image=vis_tk, bg='white')
            label.image = vis_tk
            label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать визуализацию: {e}")
            viz_window.destroy()

