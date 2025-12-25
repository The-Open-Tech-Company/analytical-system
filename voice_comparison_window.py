"""
Модуль интерфейса для сравнения двух голосовых треков
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import threading
import time
import sounddevice as sd
import soundfile as sf
from voice_analyzer import VoiceAnalyzer
from voice_comparator import VoiceComparator
from voice_visualizer import VoiceVisualizer


class VoiceComparisonWindow:
    """Класс для создания окна сравнения голосов"""
    
    def __init__(self, parent_window, main_app):
        self.parent_window = parent_window
        self.main_app = main_app
        self.parent_window.title("Сравнение голосов")
        self.parent_window.geometry("1400x900")
        self.parent_window.configure(bg='#f0f0f0')
        
        # Переменные для хранения путей к аудиофайлам
        self.audio1_path = None
        self.audio2_path = None
        self.audio1_data = None
        self.audio2_data = None
        self.sample_rate1 = None
        self.sample_rate2 = None
        self.analysis1 = None
        self.analysis2 = None
        self.comparison_result = None
        
        # Переменные для записи аудио
        self.is_recording1 = False
        self.is_recording2 = False
        self.recording_thread1 = None
        self.recording_thread2 = None
        self.recording_frames1 = []
        self.recording_frames2 = []
        self.recording_start_time1 = None
        self.recording_start_time2 = None
        self.recording_timer_id1 = None
        self.recording_timer_id2 = None
        
        # Инициализация компонентов системы
        try:
            self.analyzer = VoiceAnalyzer()
            self.comparator = VoiceComparator()
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
            text="Система сравнения голосов",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        title_label.pack(pady=10)
        
        # Основной контейнер
        main_frame = tk.Frame(self.parent_window, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - загрузка аудио
        left_panel = tk.Frame(main_frame, bg='#f0f0f0', width=600)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Панель для первого аудио
        self.create_audio_panel(left_panel, "Аудио 1", 1)
        
        # Панель для второго аудио
        self.create_audio_panel(left_panel, "Аудио 2", 2)
        
        # Правая панель - результаты
        right_panel = tk.Frame(main_frame, bg='#f0f0f0', width=600)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Кнопка сравнения
        compare_btn = tk.Button(
            right_panel,
            text="Сравнить голоса",
            font=("Arial", 14, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            command=self.compare_voices,
            cursor='hand2'
        )
        compare_btn.pack(pady=20)
        
        # Область для результатов
        results_frame = tk.LabelFrame(
            right_panel,
            text="Результаты сравнения",
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
    
    def create_audio_panel(self, parent, title, audio_num):
        """Создает панель для загрузки и отображения аудио"""
        panel = tk.LabelFrame(
            parent,
            text=title,
            font=("Arial", 11, "bold"),
            bg='#f0f0f0',
            fg='#333',
            padx=8,
            pady=8
        )
        panel.pack(fill=tk.BOTH, expand=True, pady=3)
        
        # Фрейм для кнопок
        buttons_frame = tk.Frame(panel, bg='#f0f0f0')
        buttons_frame.pack(pady=3)
        
        # Кнопка загрузки
        load_btn = tk.Button(
            buttons_frame,
            text="📁 Загрузить",
            font=("Arial", 9),
            bg='#2196F3',
            fg='white',
            padx=10,
            pady=3,
            command=lambda: self.load_audio(audio_num),
            cursor='hand2'
        )
        load_btn.pack(side=tk.LEFT, padx=2)
        
        # Кнопка записи
        if audio_num == 1:
            self.record_btn1 = tk.Button(
                buttons_frame,
                text="🎤 Записать",
                font=("Arial", 9),
                bg='#F44336',
                fg='white',
                padx=10,
                pady=3,
                command=lambda: self.toggle_recording(1),
                cursor='hand2'
            )
            self.record_btn1.pack(side=tk.LEFT, padx=2)
        else:
            self.record_btn2 = tk.Button(
                buttons_frame,
                text="🎤 Записать",
                font=("Arial", 9),
                bg='#F44336',
                fg='white',
                padx=10,
                pady=3,
                command=lambda: self.toggle_recording(2),
                cursor='hand2'
            )
            self.record_btn2.pack(side=tk.LEFT, padx=2)
        
        # Индикатор записи
        if audio_num == 1:
            self.recording_label1 = tk.Label(
                panel,
                text="",
                font=("Arial", 10, "bold"),
                bg='#f0f0f0',
                fg='red'
            )
            self.recording_label1.pack(pady=2)
            
            self.recording_progress1 = ttk.Progressbar(
                panel,
                mode='indeterminate',
                length=150
            )
            self.recording_progress1.pack(pady=2)
        else:
            self.recording_label2 = tk.Label(
                panel,
                text="",
                font=("Arial", 10, "bold"),
                bg='#f0f0f0',
                fg='red'
            )
            self.recording_label2.pack(pady=2)
            
            self.recording_progress2 = ttk.Progressbar(
                panel,
                mode='indeterminate',
                length=150
            )
            self.recording_progress2.pack(pady=2)
        
        # Метка для отображения информации
        info_label = tk.Label(
            panel,
            text="Аудио не загружено",
            bg='white',
            width=35,
            height=6,
            relief=tk.SUNKEN,
            borderwidth=1,
            font=("Arial", 9),
            justify=tk.LEFT,
            anchor='nw',
            padx=5,
            pady=5
        )
        info_label.pack(pady=3, padx=3, fill=tk.BOTH, expand=True)
        
        # Сохраняем ссылку на метку
        if audio_num == 1:
            self.audio1_info_label = info_label
        else:
            self.audio2_info_label = info_label
    
    def load_audio(self, audio_num):
        """Загружает аудиофайл"""
        file_path = filedialog.askopenfilename(
            title=f"Выберите аудиофайл {audio_num}",
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
            audio_data, sample_rate = self.analyzer.load_audio(file_path)
            duration = len(audio_data) / sample_rate
            
            if audio_num == 1:
                self.audio1_path = file_path
                self.audio1_data = audio_data
                self.sample_rate1 = sample_rate
                self.analysis1 = None
                
                info = f"Файл: {os.path.basename(file_path)}\n"
                info += f"Длительность: {duration:.2f} сек\n"
                info += f"Частота: {sample_rate} Hz\n"
                info += f"Сэмплов: {len(audio_data)}"
                
                self.audio1_info_label.config(text=info)
            else:
                self.audio2_path = file_path
                self.audio2_data = audio_data
                self.sample_rate2 = sample_rate
                self.analysis2 = None
                
                info = f"Файл: {os.path.basename(file_path)}\n"
                info += f"Длительность: {duration:.2f} сек\n"
                info += f"Частота: {sample_rate} Hz\n"
                info += f"Сэмплов: {len(audio_data)}"
                
                self.audio2_info_label.config(text=info)
            
            self.comparison_result = None
            self.clear_results()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить аудиофайл:\n{e}")
    
    def toggle_recording(self, audio_num):
        """Переключает режим записи"""
        if audio_num == 1:
            if not self.is_recording1:
                self.start_recording(1)
            else:
                self.stop_recording(1)
        else:
            if not self.is_recording2:
                self.start_recording(2)
            else:
                self.stop_recording(2)
    
    def start_recording(self, audio_num):
        """Начинает запись"""
        if audio_num == 1:
            self.is_recording1 = True
            self.recording_frames1 = []
            self.recording_start_time1 = None
            self.record_btn1.config(text="⏹ Остановить", bg='#4CAF50')
            self.recording_label1.config(text="Запись: 00:00", fg='red')
            self.recording_progress1.start(10)
            
            self.recording_thread1 = threading.Thread(target=lambda: self._record_audio(1))
            self.recording_thread1.daemon = True
            self.recording_thread1.start()
            
            self._update_recording_timer(1)
        else:
            self.is_recording2 = True
            self.recording_frames2 = []
            self.recording_start_time2 = None
            self.record_btn2.config(text="⏹ Остановить", bg='#4CAF50')
            self.recording_label2.config(text="Запись: 00:00", fg='red')
            self.recording_progress2.start(10)
            
            self.recording_thread2 = threading.Thread(target=lambda: self._record_audio(2))
            self.recording_thread2.daemon = True
            self.recording_thread2.start()
            
            self._update_recording_timer(2)
    
    def stop_recording(self, audio_num):
        """Останавливает запись"""
        if audio_num == 1:
            self.is_recording1 = False
            self.recording_progress1.stop()
            
            if self.recording_timer_id1:
                self.parent_window.after_cancel(self.recording_timer_id1)
                self.recording_timer_id1 = None
            
            self.record_btn1.config(text="🎤 Записать", bg='#F44336')
            
            if self.recording_thread1 and self.recording_thread1.is_alive():
                self.recording_thread1.join(timeout=2.0)
            
            if self.recording_frames1:
                elapsed_time = time.time() - self.recording_start_time1 if self.recording_start_time1 else 0
                
                if self.audio1_data is None and self.recording_frames1:
                    try:
                        self.audio1_data = np.concatenate(self.recording_frames1)
                        self.audio1_path = None
                    except Exception as e:
                        print(f"Ошибка объединения фреймов: {e}")
                
                self.recording_label1.config(text=f"Запись завершена ({elapsed_time:.1f} сек)", fg='green')
                
                if self.audio1_data is not None and self.sample_rate1 is not None:
                    duration = len(self.audio1_data) / self.sample_rate1
                else:
                    duration = elapsed_time
                
                info = f"Запись\n"
                info += f"Длительность: {duration:.2f} сек\n"
                info += f"Частота: {self.sample_rate1 if self.sample_rate1 else 44100} Hz\n"
                if self.audio1_data is not None:
                    info += f"Сэмплов: {len(self.audio1_data)}"
                
                self.audio1_info_label.config(text=info)
                self.analysis1 = None
                self.comparison_result = None
                self.clear_results()
            else:
                self.recording_label1.config(text="Запись не началась", fg='orange')
        else:
            self.is_recording2 = False
            self.recording_progress2.stop()
            
            if self.recording_timer_id2:
                self.parent_window.after_cancel(self.recording_timer_id2)
                self.recording_timer_id2 = None
            
            self.record_btn2.config(text="🎤 Записать", bg='#F44336')
            
            if self.recording_thread2 and self.recording_thread2.is_alive():
                self.recording_thread2.join(timeout=2.0)
            
            if self.recording_frames2:
                elapsed_time = time.time() - self.recording_start_time2 if self.recording_start_time2 else 0
                
                if self.audio2_data is None and self.recording_frames2:
                    try:
                        self.audio2_data = np.concatenate(self.recording_frames2)
                        self.audio2_path = None
                    except Exception as e:
                        print(f"Ошибка объединения фреймов: {e}")
                
                self.recording_label2.config(text=f"Запись завершена ({elapsed_time:.1f} сек)", fg='green')
                
                if self.audio2_data is not None and self.sample_rate2 is not None:
                    duration = len(self.audio2_data) / self.sample_rate2
                else:
                    duration = elapsed_time
                
                info = f"Запись\n"
                info += f"Длительность: {duration:.2f} сек\n"
                info += f"Частота: {self.sample_rate2 if self.sample_rate2 else 44100} Hz\n"
                if self.audio2_data is not None:
                    info += f"Сэмплов: {len(self.audio2_data)}"
                
                self.audio2_info_label.config(text=info)
                self.analysis2 = None
                self.comparison_result = None
                self.clear_results()
            else:
                self.recording_label2.config(text="Запись не началась", fg='orange')
    
    def _update_recording_timer(self, audio_num):
        """Обновляет таймер записи"""
        if audio_num == 1:
            if not self.is_recording1:
                return
            
            if self.recording_start_time1:
                elapsed = time.time() - self.recording_start_time1
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                self.recording_label1.config(text=f"Запись: {minutes:02d}:{seconds:02d}")
            else:
                self.recording_label1.config(text="Запись: 00:00")
            
            self.recording_timer_id1 = self.parent_window.after(1000, lambda: self._update_recording_timer(1))
        else:
            if not self.is_recording2:
                return
            
            if self.recording_start_time2:
                elapsed = time.time() - self.recording_start_time2
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                self.recording_label2.config(text=f"Запись: {minutes:02d}:{seconds:02d}")
            else:
                self.recording_label2.config(text="Запись: 00:00")
            
            self.recording_timer_id2 = self.parent_window.after(1000, lambda: self._update_recording_timer(2))
    
    def _record_audio(self, audio_num):
        """Записывает аудио (неограниченная запись)"""
        try:
            sample_rate = 44100
            chunk_size = 1024
            
            if audio_num == 1:
                self.recording_start_time1 = time.time()
                self.sample_rate1 = sample_rate
                
                with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=chunk_size) as stream:
                    while self.is_recording1:
                        chunk, overflowed = stream.read(chunk_size)
                        if overflowed:
                            print("Предупреждение: переполнение буфера (аудио 1)")
                        self.recording_frames1.append(chunk.flatten())
                
                if self.recording_frames1:
                    self.audio1_data = np.concatenate(self.recording_frames1)
                    self.audio1_path = None
                else:
                    self.audio1_data = None
            else:
                self.recording_start_time2 = time.time()
                self.sample_rate2 = sample_rate
                
                with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=chunk_size) as stream:
                    while self.is_recording2:
                        chunk, overflowed = stream.read(chunk_size)
                        if overflowed:
                            print("Предупреждение: переполнение буфера (аудио 2)")
                        self.recording_frames2.append(chunk.flatten())
                
                if self.recording_frames2:
                    self.audio2_data = np.concatenate(self.recording_frames2)
                    self.audio2_path = None
                else:
                    self.audio2_data = None
                    
        except Exception as e:
            self.parent_window.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка записи (аудио {audio_num}): {e}"))
            if audio_num == 1:
                self.is_recording1 = False
            else:
                self.is_recording2 = False
    
    def compare_voices(self):
        """Выполняет сравнение голосов"""
        # Проверяем, что оба аудио загружены (либо файлы, либо записи)
        if (self.audio1_data is None or self.audio2_data is None):
            messagebox.showwarning(
                "Предупреждение",
                "Пожалуйста, загрузите или запишите оба аудио для сравнения."
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
            # Сохраняем временные файлы, если это записи
            temp_file1 = None
            temp_file2 = None
            
            if self.audio1_path is None:
                temp_file1 = "temp_recording1.wav"
                sf.write(temp_file1, self.audio1_data, self.sample_rate1)
                file1_to_analyze = temp_file1
            else:
                file1_to_analyze = self.audio1_path
            
            if self.audio2_path is None:
                temp_file2 = "temp_recording2.wav"
                sf.write(temp_file2, self.audio2_data, self.sample_rate2)
                file2_to_analyze = temp_file2
            else:
                file2_to_analyze = self.audio2_path
            
            loading_label.config(text="Анализ первого голоса...")
            self.parent_window.update()
            
            self.analysis1 = self.analyzer.analyze_voice(file1_to_analyze)
            
            loading_label.config(text="Анализ второго голоса...")
            self.parent_window.update()
            
            self.analysis2 = self.analyzer.analyze_voice(file2_to_analyze)
            
            loading_label.config(text="Сравнение голосов...")
            self.parent_window.update()
            
            self.comparison_result = self.comparator.compare_voices(
                file1_to_analyze,
                file2_to_analyze
            )
            
            # Удаляем временные файлы
            if temp_file1 and os.path.exists(temp_file1):
                try:
                    os.remove(temp_file1)
                except:
                    pass
            if temp_file2 and os.path.exists(temp_file2):
                try:
                    os.remove(temp_file2)
                except:
                    pass
            
            loading_label.destroy()
            self.display_results()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при сравнении: {e}")
            if 'loading_label' in locals():
                loading_label.destroy()
    
    def clear_results(self):
        """Очищает область результатов"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
    
    def display_results(self):
        """Отображает результаты сравнения"""
        if not self.comparison_result:
            return
        
        title_label = tk.Label(
            self.results_frame,
            text="Результаты сравнения",
            font=("Arial", 14, "bold"),
            bg='white',
            fg='#333'
        )
        title_label.pack(pady=10)
        
        # Информация о голосах
        info_frame = tk.Frame(self.results_frame, bg='white')
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        gender1 = self.analysis1.get('gender', 'Не определен')
        gender2 = self.analysis2.get('gender', 'Не определен')
        emotion1 = self.analysis1.get('emotion', 'Не определена')
        emotion2 = self.analysis2.get('emotion', 'Не определена')
        
        info_text1 = f"Голос 1: {gender1}, {emotion1}"
        info_text2 = f"Голос 2: {gender2}, {emotion2}"
        
        info_label1 = tk.Label(
            info_frame,
            text=info_text1,
            font=("Arial", 10),
            bg='white',
            fg='#333',
            anchor='w'
        )
        info_label1.pack(fill=tk.X, pady=2)
        
        info_label2 = tk.Label(
            info_frame,
            text=info_text2,
            font=("Arial", 10),
            bg='white',
            fg='#333',
            anchor='w'
        )
        info_label2.pack(fill=tk.X, pady=2)
        
        # Разделитель
        separator = tk.Frame(self.results_frame, height=2, bg='#ccc')
        separator.pack(fill=tk.X, padx=10, pady=10)
        
        # Результаты сравнения характеристик
        comparison = self.comparison_result.get('comparison', {})
        
        feature_names_ru = {
            'pitch': 'Основная частота (Pitch)',
            'pitch_variation': 'Вариативность Pitch',
            'mfcc': 'MFCC (Тембр)',
            'spectral_centroid': 'Спектральный центроид',
            'formant_f1': 'Форманта F1',
            'formant_f2': 'Форманта F2',
            'energy': 'Энергия (RMS)',
            'zcr': 'Zero Crossing Rate',
            'speech_rate': 'Темп речи'
        }
        
        for feature_name, similarity in comparison.items():
            if feature_name == 'overall':
                continue
            
            name_ru = feature_names_ru.get(feature_name, feature_name)
            
            feature_frame = tk.Frame(self.results_frame, bg='white')
            feature_frame.pack(fill=tk.X, padx=10, pady=2)
            
            name_label = tk.Label(
                feature_frame,
                text=name_ru,
                font=("Arial", 10),
                bg='white',
                width=25,
                anchor='w'
            )
            name_label.pack(side=tk.LEFT, padx=5)
            
            progress = ttk.Progressbar(
                feature_frame,
                length=200,
                mode='determinate',
                maximum=100
            )
            progress['value'] = similarity
            progress.pack(side=tk.LEFT, padx=5)
            
            color = self.get_color_for_percentage(similarity)
            percent_label = tk.Label(
                feature_frame,
                text=f"{similarity:.1f}%",
                font=("Arial", 10, "bold"),
                bg='white',
                fg=color,
                width=8
            )
            percent_label.pack(side=tk.LEFT, padx=5)
        
        separator2 = tk.Frame(self.results_frame, height=2, bg='#ccc')
        separator2.pack(fill=tk.X, padx=10, pady=10)
        
        # Общее совпадение
        overall = comparison.get('overall', 0.0)
        overall_frame = tk.Frame(self.results_frame, bg='white')
        overall_frame.pack(fill=tk.X, padx=10, pady=5)
        
        overall_name_label = tk.Label(
            overall_frame,
            text="ОБЩЕЕ СОВПАДЕНИЕ",
            font=("Arial", 12, "bold"),
            bg='white',
            width=25,
            anchor='w'
        )
        overall_name_label.pack(side=tk.LEFT, padx=5)
        
        overall_progress = ttk.Progressbar(
            overall_frame,
            length=200,
            mode='determinate',
            maximum=100
        )
        overall_progress['value'] = overall
        overall_progress.pack(side=tk.LEFT, padx=5)
        
        overall_color = self.get_color_for_percentage(overall)
        overall_percent_label = tk.Label(
            overall_frame,
            text=f"{overall:.1f}%",
            font=("Arial", 12, "bold"),
            bg='white',
            fg=overall_color,
            width=8
        )
        overall_percent_label.pack(side=tk.LEFT, padx=5)
        
        # Кнопка визуализации
        self.create_visualization_button()
        
        self.results_canvas.update_idletasks()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        self.results_canvas.yview_moveto(0)
    
    def get_color_for_percentage(self, percentage):
        """Возвращает цвет в зависимости от процента совпадения"""
        if percentage >= 80:
            return '#4CAF50'
        elif percentage >= 60:
            return '#FF9800'
        else:
            return '#F44336'
    
    def create_visualization_button(self):
        """Создает кнопку для просмотра визуализаций"""
        viz_frame = tk.LabelFrame(
            self.results_frame,
            text="Действия",
            font=("Arial", 11, "bold"),
            bg='white',
            fg='#333',
            padx=10,
            pady=10
        )
        viz_frame.pack(fill=tk.X, padx=10, pady=15)
        
        btn = tk.Button(
            viz_frame,
            text="📊 Просмотр визуализаций",
            font=("Arial", 11, "bold"),
            bg='#2196F3',
            fg='white',
            padx=20,
            pady=10,
            command=self.show_visualizations,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=2
        )
        btn.pack(expand=True)
    
    def show_visualizations(self):
        """Открывает окно с визуализациями"""
        if (self.comparison_result is None or 
            self.audio1_data is None or len(self.audio1_data) == 0 or
            self.audio2_data is None or len(self.audio2_data) == 0):
            messagebox.showwarning("Предупреждение", "Сначала выполните сравнение голосов.")
            return
        
        # Создаем новое окно для визуализаций
        viz_window = tk.Toplevel(self.parent_window)
        viz_window.transient(self.parent_window)
        viz_window.title("Визуализации сравнения голосов")
        viz_window.geometry("1600x1200")
        viz_window.configure(bg='#f0f0f0')
        
        try:
            features1 = self.analysis1.get('features', {})
            features2 = self.analysis2.get('features', {})
            
            # Создаем визуализации для обоих голосов
            vis1 = self.visualizer.create_comprehensive_visualization(
                self.audio1_data,
                self.sample_rate1,
                features1
            )
            
            vis2 = self.visualizer.create_comprehensive_visualization(
                self.audio2_data,
                self.sample_rate2,
                features2
            )
            
            # Конвертируем в PIL Image
            vis1_pil = Image.fromarray(vis1)
            vis2_pil = Image.fromarray(vis2)
            
            # Масштабируем для отображения
            max_width = 700
            vis1_pil.thumbnail((max_width, max_width), Image.Resampling.LANCZOS)
            vis2_pil.thumbnail((max_width, max_width), Image.Resampling.LANCZOS)
            
            vis1_tk = ImageTk.PhotoImage(vis1_pil)
            vis2_tk = ImageTk.PhotoImage(vis2_pil)
            
            # Основной контейнер с прокруткой
            canvas = tk.Canvas(viz_window, bg='#f0f0f0', highlightthickness=0)
            scrollbar = ttk.Scrollbar(viz_window, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Визуализация первого голоса
            frame1 = tk.LabelFrame(
                scrollable_frame,
                text="Голос 1",
                font=("Arial", 12, "bold"),
                bg='#f0f0f0',
                padx=10,
                pady=10
            )
            frame1.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
            
            label1 = tk.Label(frame1, image=vis1_tk, bg='white')
            label1.image = vis1_tk
            label1.pack(padx=10, pady=10)
            
            # Визуализация второго голоса
            frame2 = tk.LabelFrame(
                scrollable_frame,
                text="Голос 2",
                font=("Arial", 12, "bold"),
                bg='#f0f0f0',
                padx=10,
                pady=10
            )
            frame2.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
            
            label2 = tk.Label(frame2, image=vis2_tk, bg='white')
            label2.image = vis2_tk
            label2.pack(padx=10, pady=10)
            
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать визуализации: {e}")
            viz_window.destroy()

