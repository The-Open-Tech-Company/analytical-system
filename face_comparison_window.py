"""
Модуль интерфейса для сравнения двух лиц
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from face_analyzer import FaceAnalyzer
from face_comparator import FaceComparator
from face_visualizer import FaceVisualizer


class FaceComparisonWindow:
    """Класс для создания окна сравнения лиц"""
    
    def __init__(self, parent_window, main_app):
        self.parent_window = parent_window
        self.main_app = main_app
        self.parent_window.title("Идентификация личности - Сравнение лиц")
        self.parent_window.geometry("1400x900")
        self.parent_window.configure(bg='#f0f0f0')
        
        # Переменные для хранения путей к изображениям
        self.image1_path = None
        self.image2_path = None
        self.features1 = None
        self.features2 = None
        self.results = None
        
        # Переменные для поворота изображений
        self.image1_rotation = 0
        self.image2_rotation = 0
        self.original_image1 = None
        self.original_image2 = None
        
        # Инициализация компонентов системы
        try:
            self.analyzer = FaceAnalyzer()
            self.comparator = FaceComparator()
            self.visualizer = FaceVisualizer()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка инициализации системы: {e}")
            return
        
        self.create_widgets()
        
        # Привязываем обработчики клавиатуры для поворота стрелками
        self.parent_window.bind('<Left>', self.on_arrow_key)
        self.parent_window.bind('<Right>', self.on_arrow_key)
        self.parent_window.bind('<Up>', self.on_arrow_key)
        self.parent_window.bind('<Down>', self.on_arrow_key)
        self.parent_window.focus_set()
    
    def on_arrow_key(self, event):
        """Обработчик нажатий стрелок для поворота изображений"""
        active = getattr(self, 'active_image', 1)
        
        if event.keysym == 'Left':
            if active == 1 and self.original_image1 is not None:
                self.rotate_image(1, -5)
            elif active == 2 and self.original_image2 is not None:
                self.rotate_image(2, -5)
        elif event.keysym == 'Right':
            if active == 1 and self.original_image1 is not None:
                self.rotate_image(1, 5)
            elif active == 2 and self.original_image2 is not None:
                self.rotate_image(2, 5)
        elif event.keysym == 'Up':
            if active == 1 and self.original_image1 is not None:
                self.rotate_image(1, -1)
            elif active == 2 and self.original_image2 is not None:
                self.rotate_image(2, -1)
        elif event.keysym == 'Down':
            if active == 1 and self.original_image1 is not None:
                self.rotate_image(1, 1)
            elif active == 2 and self.original_image2 is not None:
                self.rotate_image(2, 1)
    
    def create_widgets(self):
        """Создает виджеты интерфейса"""
        
        # Заголовок
        title_label = tk.Label(
            self.parent_window, 
            text="Система идентификации личности",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        title_label.pack(pady=10)
        
        # Подзаголовок с информацией о строгости
        subtitle_label = tk.Label(
            self.parent_window,
            text="Строгий режим идентификации: система определяет, тот же это человек или нет",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#666'
        )
        subtitle_label.pack(pady=5)
        
        # Основной контейнер
        main_frame = tk.Frame(self.parent_window, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - загрузка изображений
        left_panel = tk.Frame(main_frame, bg='#f0f0f0', width=600)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Панель для первого изображения
        self.create_image_panel(left_panel, "Изображение 1", 1)
        
        # Панель для второго изображения
        self.create_image_panel(left_panel, "Изображение 2", 2)
        
        # Правая панель - результаты
        right_panel = tk.Frame(main_frame, bg='#f0f0f0', width=600)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Кнопка сравнения
        compare_btn = tk.Button(
            right_panel,
            text="Идентифицировать личность",
            font=("Arial", 14, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            command=self.compare_faces,
            cursor='hand2'
        )
        compare_btn.pack(pady=20)
        
        # Область для результатов
        results_frame = tk.LabelFrame(
            right_panel,
            text="Результаты идентификации личности",
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
    
    def get_current_image(self, image_num):
        """Получает текущее изображение с учетом поворота"""
        if image_num == 1:
            if self.original_image1 is None:
                if self.image1_path:
                    return cv2.imread(self.image1_path)
                return None
            return self.apply_rotation(self.original_image1, self.image1_rotation)
        else:
            if self.original_image2 is None:
                if self.image2_path:
                    return cv2.imread(self.image2_path)
                return None
            return self.apply_rotation(self.original_image2, self.image2_rotation)
    
    def create_image_panel(self, parent, title, image_num):
        """Создает панель для загрузки и отображения изображения"""
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
        
        # Панель кнопок
        buttons_frame = tk.Frame(panel, bg='#f0f0f0')
        buttons_frame.pack(pady=3)
        
        # Кнопка загрузки
        btn = tk.Button(
            buttons_frame,
            text="📁 Загрузить",
            font=("Arial", 9),
            bg='#2196F3',
            fg='white',
            padx=10,
            pady=3,
            command=lambda: self.load_image(image_num),
            cursor='hand2'
        )
        btn.pack(side=tk.LEFT, padx=2)
        
        # Кнопки поворота
        rotate_left_btn = tk.Button(
            buttons_frame,
            text="↶ -5°",
            font=("Arial", 8),
            bg='#FF9800',
            fg='white',
            padx=5,
            pady=3,
            command=lambda: self.rotate_image(image_num, -5),
            cursor='hand2'
        )
        rotate_left_btn.pack(side=tk.LEFT, padx=2)
        
        rotate_right_btn = tk.Button(
            buttons_frame,
            text="↷ +5°",
            font=("Arial", 8),
            bg='#FF9800',
            fg='white',
            padx=5,
            pady=3,
            command=lambda: self.rotate_image(image_num, 5),
            cursor='hand2'
        )
        rotate_right_btn.pack(side=tk.LEFT, padx=2)
        
        reset_btn = tk.Button(
            buttons_frame,
            text="↻ Сброс",
            font=("Arial", 8),
            bg='#9E9E9E',
            fg='white',
            padx=5,
            pady=3,
            command=lambda: self.reset_rotation(image_num),
            cursor='hand2'
        )
        reset_btn.pack(side=tk.LEFT, padx=2)
        
        # Метка для отображения изображения
        image_label = tk.Label(
            panel,
            text="Изображение не загружено",
            bg='white',
            width=35,
            height=12,
            relief=tk.SUNKEN,
            borderwidth=1
        )
        image_label.pack(pady=3, padx=3, fill=tk.BOTH, expand=True)
        
        # Сохраняем ссылку на метку
        if image_num == 1:
            self.image1_label = image_label
            image_label.bind('<Button-1>', lambda e: self.set_active_image(1))
        else:
            self.image2_label = image_label
            image_label.bind('<Button-1>', lambda e: self.set_active_image(2))
        
        if not hasattr(self, 'active_image'):
            self.active_image = 1
    
    def set_active_image(self, image_num):
        """Устанавливает активное изображение для поворота стрелками"""
        self.active_image = image_num
        self.parent_window.focus_set()
    
    def load_image(self, image_num):
        """Загружает изображение"""
        file_path = filedialog.askopenfilename(
            title=f"Выберите изображение {image_num}",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Все файлы", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Ошибка", f"Файл не найден:\n{file_path}")
            return
        
        try:
            img = None
            
            # Попытка 1: стандартная загрузка через OpenCV
            img = cv2.imread(file_path)
            
            # Попытка 2: через numpy
            if img is None:
                try:
                    with open(file_path, 'rb') as f:
                        image_bytes = f.read()
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception as e:
                    pass
            
            # Попытка 3: через PIL
            if img is None:
                try:
                    pil_img = Image.open(file_path)
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    img_array = np.array(pil_img)
                    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    pass
            
            if img is None:
                raise ValueError(
                    f"Не удалось загрузить изображение.\n"
                    f"Проверьте:\n"
                    f"- Поддерживается ли формат файла (JPG, PNG, BMP, GIF)\n"
                    f"- Не поврежден ли файл\n"
                    f"- Доступен ли файл для чтения"
                )
            
            if img.size == 0:
                raise ValueError("Изображение пустое или повреждено")
            
            if image_num == 1:
                self.image1_path = file_path
                self.original_image1 = img.copy()
                self.image1_rotation = 0
            else:
                self.image2_path = file_path
                self.original_image2 = img.copy()
                self.image2_rotation = 0
            
            self.display_image(image_num, img)
            self.results = None
            self.clear_results()
            
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror(
                "Ошибка загрузки изображения",
                f"Не удалось загрузить изображение:\n\n{error_msg}\n\n"
                f"Путь к файлу: {file_path}"
            )
    
    def rotate_image(self, image_num, angle):
        """Поворачивает изображение на указанный угол"""
        if image_num == 1:
            if self.original_image1 is None:
                return
            self.image1_rotation += angle
            rotated = self.apply_rotation(self.original_image1, self.image1_rotation)
            self.display_image(1, rotated)
            self.image1_path = None
        else:
            if self.original_image2 is None:
                return
            self.image2_rotation += angle
            rotated = self.apply_rotation(self.original_image2, self.image2_rotation)
            self.display_image(2, rotated)
            self.image2_path = None
        
        self.results = None
        self.clear_results()
    
    def reset_rotation(self, image_num):
        """Сбрасывает поворот изображения"""
        if image_num == 1:
            if self.original_image1 is None:
                return
            self.image1_rotation = 0
            self.display_image(1, self.original_image1)
            self.image1_path = None
        else:
            if self.original_image2 is None:
                return
            self.image2_rotation = 0
            self.display_image(2, self.original_image2)
            self.image2_path = None
        
        self.results = None
        self.clear_results()
    
    def apply_rotation(self, image, angle):
        """Применяет поворот к изображению"""
        if abs(angle) < 0.1:
            return image.copy()
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        return rotated
    
    def display_image(self, image_num, image):
        """Отображает изображение в панели"""
        try:
            if image is None or image.size == 0:
                raise ValueError("Изображение пустое")
            
            if len(image.shape) == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((280, 280), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_pil)
            
            if image_num == 1:
                self.image1_label.configure(image=photo, text="")
                self.image1_label.image = photo
            else:
                self.image2_label.configure(image=photo, text="")
                self.image2_label.image = photo
        except Exception as e:
            error_msg = f"Ошибка при отображении изображения: {e}"
            if image_num == 1:
                self.image1_label.configure(text=error_msg, image="")
            else:
                self.image2_label.configure(text=error_msg, image="")
            messagebox.showerror("Ошибка отображения", error_msg)
    
    def compare_faces(self):
        """Выполняет сравнение лиц"""
        img1 = self.get_current_image(1)
        img2 = self.get_current_image(2)
        
        if img1 is None or img2 is None:
            messagebox.showwarning(
                "Предупреждение",
                "Пожалуйста, загрузите оба изображения для сравнения."
            )
            return
        
        self.clear_results()
        
        loading_label = tk.Label(
            self.results_frame,
            text="Обработка изображений...",
            font=("Arial", 12),
            bg='white'
        )
        loading_label.pack(pady=20)
        self.parent_window.update()
        
        try:
            img1 = self.get_current_image(1)
            img2 = self.get_current_image(2)
            
            temp_path1 = "temp_image1.jpg"
            temp_path2 = "temp_image2.jpg"
            cv2.imwrite(temp_path1, img1)
            cv2.imwrite(temp_path2, img2)
            
            loading_label.config(text="Извлечение характеристик из первого изображения...")
            self.parent_window.update()
            self.features1 = self.analyzer.extract_face_features(temp_path1)
            
            if self.features1 is None:
                messagebox.showerror(
                    "Ошибка",
                    "Лицо не найдено на первом изображении!\n"
                    "Убедитесь, что на изображении четко видно одно лицо (анфас)."
                )
                loading_label.destroy()
                return
            
            loading_label.config(text="Извлечение характеристик из второго изображения...")
            self.parent_window.update()
            self.features2 = self.analyzer.extract_face_features(temp_path2)
            
            try:
                os.remove(temp_path1)
                os.remove(temp_path2)
            except:
                pass
            
            if self.features2 is None:
                messagebox.showerror(
                    "Ошибка",
                    "Лицо не найдено на втором изображении!\n"
                    "Убедитесь, что на изображении четко видно одно лицо (анфас)."
                )
                loading_label.destroy()
                return
            
            loading_label.config(text="Сравнение лиц...")
            self.parent_window.update()
            self.results = self.comparator.compare_faces(self.features1, self.features2)
            
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
        if not self.results:
            return
        
        title_label = tk.Label(
            self.results_frame,
            text="Результаты идентификации личности",
            font=("Arial", 14, "bold"),
            bg='white',
            fg='#333'
        )
        title_label.pack(pady=10)
        
        info_frame = tk.Frame(self.results_frame, bg='white')
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        gender1 = self.features1.get('gender', 'Не определен')
        age1 = self.features1.get('age', 'Не определен')
        race1 = self.features1.get('race', 'Не определена')
        gender2 = self.features2.get('gender', 'Не определен')
        age2 = self.features2.get('age', 'Не определен')
        race2 = self.features2.get('race', 'Не определена')
        
        gender_conf1 = self.features1.get('gender_confidence', 0.0)
        age_conf1 = self.features1.get('age_confidence', 0.0)
        race_conf1 = self.features1.get('race_confidence', 0.0)
        gender_conf2 = self.features2.get('gender_confidence', 0.0)
        age_conf2 = self.features2.get('age_confidence', 0.0)
        race_conf2 = self.features2.get('race_confidence', 0.0)
        
        # Убеждаемся, что уверенность в диапазоне [0, 1] и положительная
        gender_conf1 = max(0.0, min(1.0, abs(float(gender_conf1))))
        age_conf1 = max(0.0, min(1.0, abs(float(age_conf1))))
        race_conf1 = max(0.0, min(1.0, abs(float(race_conf1))))
        gender_conf2 = max(0.0, min(1.0, abs(float(gender_conf2))))
        age_conf2 = max(0.0, min(1.0, abs(float(age_conf2))))
        race_conf2 = max(0.0, min(1.0, abs(float(race_conf2))))
        
        info_text1 = f"Лицо 1: {gender1} (уверенность: {gender_conf1*100:.0f}%), {age1} (уверенность: {age_conf1*100:.0f}%), {race1} (уверенность: {race_conf1*100:.0f}%)"
        info_text2 = f"Лицо 2: {gender2} (уверенность: {gender_conf2*100:.0f}%), {age2} (уверенность: {age_conf2*100:.0f}%), {race2} (уверенность: {race_conf2*100:.0f}%)"
        
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
        
        # Предупреждение, если пол разный
        if (gender1 != 'Не определен' and gender2 != 'Не определен' and 
            gender1 != gender2):
            warning_text = f"⚠️ ВНИМАНИЕ: Пол разный ({gender1} vs {gender2}). "
            if gender_conf1 > 0.7 and gender_conf2 > 0.7:
                warning_text += "Это точно разные люди. Совпадение установлено в 0%."
            else:
                warning_text += "Совпадение снижено из-за разного пола."
            
            warning_label = tk.Label(
                info_frame,
                text=warning_text,
                font=("Arial", 10, "bold"),
                bg='#fff3cd',
                fg='#856404',
                anchor='w',
                wraplength=500
            )
            warning_label.pack(fill=tk.X, pady=5, padx=5)
        
        feature_names_ru = {
            'face_oval': 'Овал лица',
            'head_shape': 'Форма головы',
            'left_eye': 'Левый глаз',
            'right_eye': 'Правый глаз',
            'left_eyebrow': 'Левая бровь',
            'right_eyebrow': 'Правая бровь',
            'nose_bridge': 'Спинка носа',
            'nose_tip': 'Кончик носа',
            'nose_contour': 'Контур носа',
            'mouth_outer': 'Рот (внешний)',
            'mouth_inner': 'Рот (внутренний)',
            'upper_lip': 'Верхняя губа',
            'lower_lip': 'Нижняя губа',
            'left_cheek': 'Левая скула',
            'right_cheek': 'Правая скула',
            'left_ear': 'Левое ухо',
            'right_ear': 'Правое ухо',
            'left_ear_detail': 'Левое ухо (детали)',
            'right_ear_detail': 'Правое ухо (детали)',
            'chin': 'Подбородок',
            'forehead': 'Лоб',
            'hair': 'Волосы',
            'hairline': 'Линия роста волос',
            'left_temple': 'Левый висок',
            'right_temple': 'Правый висок'
        }
        
        for feature_name, similarity in self.results.items():
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
                width=20,
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
        
        separator = tk.Frame(self.results_frame, height=2, bg='#ccc')
        separator.pack(fill=tk.X, padx=10, pady=10)
        
        overall = self.results.get('overall', 0.0)
        overall_frame = tk.Frame(self.results_frame, bg='white')
        overall_frame.pack(fill=tk.X, padx=10, pady=5)
        
        overall_name_label = tk.Label(
            overall_frame,
            text="ВЕРОЯТНОСТЬ ИДЕНТИЧНОСТИ",
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
        
        self.create_visualization_buttons()
        
        self.results_canvas.update_idletasks()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        self.results_canvas.yview_moveto(0)
    
    def get_color_for_percentage(self, percentage):
        """Возвращает цвет в зависимости от процента совпадения
        УЖЕСТОЧЕНО для идентификации личности: зеленый только для очень высоких процентов
        """
        if percentage >= 85:
            return '#4CAF50'  # Зеленый - высокая вероятность идентичности
        elif percentage >= 70:
            return '#FF9800'  # Оранжевый - средняя вероятность
        elif percentage >= 50:
            return '#FFC107'  # Желтый - низкая вероятность
        else:
            return '#F44336'  # Красный - очень низкая вероятность (разные люди)
    
    def create_visualization_buttons(self):
        """Создает кнопки для просмотра визуализаций"""
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
        
        buttons_container = tk.Frame(viz_frame, bg='white')
        buttons_container.pack(expand=True)
        
        btn1 = tk.Button(
            buttons_container,
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
        btn1.pack(side=tk.LEFT, padx=10)
        
        btn2 = tk.Button(
            buttons_container,
            text="💾 Сохранить результаты",
            font=("Arial", 11, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            command=self.save_results,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=2
        )
        btn2.pack(side=tk.LEFT, padx=10)
    
    def show_visualizations(self):
        """Открывает окно с визуализациями"""
        if not self.results or not self.features1 or not self.features2:
            messagebox.showwarning("Предупреждение", "Сначала выполните сравнение лиц.")
            return
        
        # Создаем новое окно для визуализаций
        viz_window = tk.Toplevel(self.parent_window)
        viz_window.transient(self.parent_window)  # Не будет перекрывать родительское окно
        viz_window.title("Визуализации сравнения")
        viz_window.geometry("1400x900")
        viz_window.configure(bg='#f0f0f0')
        viz_window.minsize(1200, 700)
        
        # Переменные для режима отображения и поворота
        viz_mode = tk.StringVar(value="overall")
        viz_rotation1 = 0
        viz_rotation2 = 0
        selected_feature = tk.StringVar(value="")
        
        # Создаем визуализации
        try:
            image1 = self.features1['image'].copy()
            image2 = self.features2['image'].copy()
            
            vis1 = self.visualizer.visualize_face_features(
                image1, self.features1, self.visualizer.color_green
            )
            vis2 = self.visualizer.visualize_face_features(
                image2, self.features2, self.visualizer.color_red
            )
            overlay = self.visualizer.create_overlay_comparison(
                self.features1, self.features2, image1, image2, self.results
            )
            
            # Конвертируем в формат для tkinter
            vis1_rgb = cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB)
            vis2_rgb = cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            
            # Масштабируем изображения
            def resize_for_display(img, max_size=400, max_height=None):
                h, w = img.shape[:2]
                if max_height:
                    scale = min(max_size / w, max_height / h, 1.0)
                else:
                    scale = min(max_size / w, max_size / h, 1.0)
                new_w = int(w * scale)
                new_h = int(h * scale)
                return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            vis1_resized = resize_for_display(vis1_rgb, max_size=350)
            vis2_resized = resize_for_display(vis2_rgb, max_size=350)
            overlay_resized = resize_for_display(overlay_rgb, max_size=800, max_height=500)
            
            # Конвертируем в PIL Image
            vis1_pil = Image.fromarray(vis1_resized)
            vis2_pil = Image.fromarray(vis2_resized)
            overlay_pil = Image.fromarray(overlay_resized)
            
            # Конвертируем в ImageTk
            vis1_tk = ImageTk.PhotoImage(vis1_pil)
            vis2_tk = ImageTk.PhotoImage(vis2_pil)
            overlay_tk = ImageTk.PhotoImage(overlay_pil)
            
            # Панель управления
            control_frame = tk.LabelFrame(
                viz_window,
                text="Управление",
                font=("Arial", 9, "bold"),
                bg='#f0f0f0',
                padx=8,
                pady=6
            )
            control_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Основной контейнер с прокруткой
            main_viz_container = tk.Frame(viz_window, bg='#f0f0f0')
            main_viz_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            viz_canvas = tk.Canvas(main_viz_container, bg='#f0f0f0', highlightthickness=0)
            viz_scrollbar = ttk.Scrollbar(main_viz_container, orient="vertical", command=viz_canvas.yview)
            scrollable_viz_frame = tk.Frame(viz_canvas, bg='#f0f0f0')
            
            scrollable_viz_frame.bind(
                "<Configure>",
                lambda e: viz_canvas.configure(scrollregion=viz_canvas.bbox("all"))
            )
            
            viz_canvas.create_window((0, 0), window=scrollable_viz_frame, anchor="nw")
            viz_canvas.configure(yscrollcommand=viz_scrollbar.set)
            
            viz_canvas.pack(side="left", fill="both", expand=True)
            viz_scrollbar.pack(side="right", fill="y")
            
            # Область для визуализаций
            main_content_frame = tk.Frame(scrollable_viz_frame, bg='#f0f0f0')
            main_content_frame.pack(pady=5, fill=tk.BOTH, expand=True)
            
            # Левая часть - фото
            left_photos_frame = tk.Frame(main_content_frame, bg='#f0f0f0')
            left_photos_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)
            
            # Правая часть - общая визуализация
            right_overlay_frame = tk.Frame(main_content_frame, bg='#f0f0f0')
            right_overlay_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            buttons_row = tk.Frame(control_frame, bg='#f0f0f0')
            buttons_row.pack()
            
            # Переключатель режима визуализации
            mode_frame = tk.Frame(buttons_row, bg='#f0f0f0')
            mode_frame.pack(side=tk.LEFT, padx=5)
            
            tk.Label(mode_frame, text="Режим:", font=("Arial", 9), bg='#f0f0f0').pack(side=tk.LEFT, padx=2)
            
            # Словарь названий черт
            feature_names_ru = {
                'face_oval': 'Овал лица',
                'head_shape': 'Форма головы',
                'left_eye': 'Левый глаз',
                'right_eye': 'Правый глаз',
                'left_eyebrow': 'Левая бровь',
                'right_eyebrow': 'Правая бровь',
                'nose_bridge': 'Спинка носа',
                'nose_tip': 'Кончик носа',
                'nose_contour': 'Контур носа',
                'mouth_outer': 'Рот (внешний)',
                'mouth_inner': 'Рот (внутренний)',
                'upper_lip': 'Верхняя губа',
                'lower_lip': 'Нижняя губа',
                'left_cheek': 'Левая скула',
                'right_cheek': 'Правая скула',
                'left_ear': 'Левое ухо',
                'right_ear': 'Правое ухо',
                'chin': 'Подбородок',
                'forehead': 'Лоб',
                'hair': 'Волосы',
                'hairline': 'Линия роста волос',
                'left_temple': 'Левый висок',
                'right_temple': 'Правый висок'
            }
            
            # Получаем список доступных черт (хотя бы у одного лица должны быть точки)
            available_features = []
            for feat_name in feature_names_ru.keys():
                points1 = self.features1.get(feat_name, np.array([]))
                points2 = self.features2.get(feat_name, np.array([]))
                # Проверяем, что это numpy array
                if isinstance(points1, np.ndarray) and isinstance(points2, np.ndarray):
                    if len(points1) > 0 or len(points2) > 0:
                        available_features.append(feat_name)
                elif len(points1) > 0 or len(points2) > 0:
                    available_features.append(feat_name)
            
            # Создаем словарь для отображения ПЕРЕД созданием selected_feature
            feature_display_map = {feature_names_ru.get(f, f): f for f in available_features}
            
            selected_feature = tk.StringVar(value="")
            if available_features and feature_display_map:
                # Устанавливаем первое отображаемое имя, а не имя характеристики
                first_display = list(feature_display_map.keys())[0]
                selected_feature.set(first_display)
            
            # Выбор черты (только для режима "По элементам")
            feature_select_frame = tk.Frame(control_frame, bg='#f0f0f0')
            feature_select_frame.pack(fill=tk.X, padx=5, pady=5)
            
            tk.Label(feature_select_frame, text="Выберите черту лица:", font=("Arial", 9, "bold"), bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
            
            feature_combo = ttk.Combobox(
                feature_select_frame,
                textvariable=selected_feature,
                values=list(feature_display_map.keys()),
                state="readonly",
                width=25,
                font=("Arial", 9)
            )
            feature_combo.pack(side=tk.LEFT, padx=5)
            
            if available_features and feature_display_map:
                feature_combo.current(0)
                # Убеждаемся, что selected_feature содержит правильное значение
                if not selected_feature.get():
                    selected_feature.set(list(feature_display_map.keys())[0])
            
            feature_combo.bind("<<ComboboxSelected>>", lambda e: update_viz())
            
            def update_viz():
                # Показываем/скрываем выбор черты
                if viz_mode.get() == "detailed":
                    if available_features and feature_display_map:
                        # Показываем выбор черты
                        if not feature_select_frame.winfo_viewable():
                            feature_select_frame.pack(fill=tk.X, padx=5, pady=5)
                    else:
                        feature_select_frame.pack_forget()
                        # Показываем сообщение об отсутствии черт
                        for widget in left_photos_frame.winfo_children() + right_overlay_frame.winfo_children():
                            widget.destroy()
                        error_label = tk.Label(right_overlay_frame, 
                                              text="Нет доступных черт лица для отображения", 
                                              font=("Arial", 10), bg='white', fg='red')
                        error_label.pack(padx=15, pady=15)
                        return
                else:
                    feature_select_frame.pack_forget()
                
                for widget in left_photos_frame.winfo_children() + right_overlay_frame.winfo_children():
                    widget.destroy()
                
                if viz_mode.get() == "overall":
                    # Общая визуализация
                    vis1_rot = self.visualizer.visualize_face_features(
                        image1.copy(), self.features1, self.visualizer.color_green
                    )
                    vis2_rot = self.visualizer.visualize_face_features(
                        image2.copy(), self.features2, self.visualizer.color_red
                    )
                    
                    overlay_rot = self.visualizer.create_overlay_comparison(
                        self.features1, self.features2, image1.copy(), image2.copy(), self.results
                    )
                    
                    vis1_rot_rgb = cv2.cvtColor(vis1_rot, cv2.COLOR_BGR2RGB)
                    vis2_rot_rgb = cv2.cvtColor(vis2_rot, cv2.COLOR_BGR2RGB)
                    overlay_rot_rgb = cv2.cvtColor(overlay_rot, cv2.COLOR_BGR2RGB)
                    
                    vis1_rot_resized = resize_for_display(vis1_rot_rgb, max_size=300)
                    vis2_rot_resized = resize_for_display(vis2_rot_rgb, max_size=300)
                    overlay_rot_resized = resize_for_display(overlay_rot_rgb, max_size=800, max_height=500)
                    
                    vis1_rot_pil = Image.fromarray(vis1_rot_resized)
                    vis2_rot_pil = Image.fromarray(vis2_rot_resized)
                    overlay_rot_pil = Image.fromarray(overlay_rot_resized)
                    
                    vis1_rot_tk = ImageTk.PhotoImage(vis1_rot_pil)
                    vis2_rot_tk = ImageTk.PhotoImage(vis2_rot_pil)
                    overlay_rot_tk = ImageTk.PhotoImage(overlay_rot_pil)
                    
                    # ЛЕВАЯ ЧАСТЬ - фото лиц
                    photos_label = tk.Label(left_photos_frame, text="Фото для сравнения", 
                                           font=("Arial", 11, "bold"), bg='#f0f0f0', fg='#333')
                    photos_label.pack(pady=(0, 10))
                    
                    frame1 = tk.LabelFrame(left_photos_frame, text="Лицо 1 (Зеленый)", 
                                          font=("Arial", 10, "bold"), 
                                          bg='#f0f0f0', fg='#2E7D32', padx=10, pady=5)
                    frame1.pack(pady=10)
                    label1 = tk.Label(frame1, image=vis1_rot_tk, bg='white', relief=tk.RAISED, 
                                     borderwidth=2)
                    label1.image = vis1_rot_tk
                    label1.pack(padx=5, pady=5)
                    
                    frame2 = tk.LabelFrame(left_photos_frame, text="Лицо 2 (Красный)", 
                                          font=("Arial", 10, "bold"), 
                                          bg='#f0f0f0', fg='#C62828', padx=10, pady=5)
                    frame2.pack(pady=10)
                    label2 = tk.Label(frame2, image=vis2_rot_tk, bg='white', relief=tk.RAISED, 
                                     borderwidth=2)
                    label2.image = vis2_rot_tk
                    label2.pack(padx=5, pady=5)
                    
                    # ПРАВАЯ ЧАСТЬ - общая визуализация
                    frame3 = tk.LabelFrame(right_overlay_frame, 
                                          text="Общая визуализация сравнения (Зеленый - Лицо 1, Красный - Лицо 2)", 
                                          font=("Arial", 11, "bold"), bg='#f0f0f0', fg='#333', 
                                          padx=15, pady=10)
                    frame3.pack(fill=tk.BOTH, expand=True)
                    
                    overlay_inner = tk.Frame(frame3, bg='white')
                    overlay_inner.pack(expand=True, fill=tk.BOTH, padx=15, pady=15)
                    
                    label3 = tk.Label(overlay_inner, image=overlay_rot_tk, bg='white')
                    label3.image = overlay_rot_tk
                    label3.pack(expand=True)
                    
                    viz_canvas.update_idletasks()
                    viz_canvas.configure(scrollregion=viz_canvas.bbox("all"))
                else:
                    # Поэлементная визуализация - используем метод из старой версии
                    display_name = selected_feature.get()
                    
                    # Проверяем, что display_name не пустой и существует в feature_display_map
                    if not display_name or display_name not in feature_display_map:
                        if available_features and feature_display_map:
                            # Используем первое доступное значение
                            display_name = list(feature_display_map.keys())[0]
                            selected_feature.set(display_name)
                        else:
                            return
                    
                    feat_name = feature_display_map.get(display_name)
                    
                    if feat_name is None and available_features:
                        feat_name = available_features[0]
                    
                    if feat_name is None:
                        return
                    
                    # Нормализуем размеры лиц ПЕРЕД визуализацией (как в старой версии)
                    from face_comparator import FaceComparator
                    comparator = FaceComparator()
                    normalized_features1, normalized_features2 = comparator.normalize_face_size(self.features1, self.features2)
                    
                    points1 = normalized_features1.get(feat_name, np.array([]))
                    points2 = normalized_features2.get(feat_name, np.array([]))
                    
                    # Проверяем, что хотя бы у одного лица есть точки для выбранной черты
                    if len(points1) > 0 or len(points2) > 0:
                        name_ru = feature_names_ru.get(feat_name, feat_name)
                        
                        try:
                            # Убеждаемся, что points - это numpy array с правильной формой
                            if len(points1) > 0:
                                if not isinstance(points1, np.ndarray):
                                    points1 = np.array(points1, dtype=np.float64)
                                if len(points1.shape) != 2 or points1.shape[1] != 2:
                                    points1 = points1.reshape(-1, 2)
                                points1 = points1.astype(np.float64)
                            
                            if len(points2) > 0:
                                if not isinstance(points2, np.ndarray):
                                    points2 = np.array(points2, dtype=np.float64)
                                if len(points2.shape) != 2 or points2.shape[1] != 2:
                                    points2 = points2.reshape(-1, 2)
                                points2 = points2.astype(np.float64)
                            
                            def create_overlay_visualization(points1, points2, feat_name=None, target_size=500):
                                """Создает наложенную визуализацию двух нормализованных элементов на белом фоне с правильным выравниванием (метод из старой версии)"""
                                if len(points1) == 0 and len(points2) == 0:
                                    return None
                                
                                # Если у одного из лиц нет точек, используем только точки другого
                                if len(points1) == 0:
                                    points1 = points2.copy()  # Используем копию для визуализации
                                if len(points2) == 0:
                                    points2 = points1.copy()  # Используем копию для визуализации
                                
                                # Убеждаемся, что точки имеют правильную форму
                                if len(points1.shape) != 2 or points1.shape[1] != 2:
                                    return None
                                if len(points2.shape) != 2 or points2.shape[1] != 2:
                                    return None
                                
                                try:
                                    # Вычисляем центроиды для выравнивания (как в общей визуализации)
                                    centroid1 = np.mean(points1, axis=0)
                                    centroid2 = np.mean(points2, axis=0)
                                    centroid1 = np.array([float(centroid1[0]), float(centroid1[1])])
                                    centroid2 = np.array([float(centroid2[0]), float(centroid2[1])])
                                    
                                    # Вычисляем размеры элементов (уже нормализованных)
                                    size1 = np.max(points1, axis=0) - np.min(points1, axis=0)
                                    size2 = np.max(points2, axis=0) - np.min(points2, axis=0)
                                    avg_size = float((float(np.mean(size1)) + float(np.mean(size2))) / 2.0)
                                    
                                    # Определяем размер канваса
                                    canvas_size = target_size
                                    
                                    # Создаем белый фон (только белый, без изображений)
                                    white_bg = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
                                    
                                    # Вычисляем масштаб для нормализации
                                    # Используем одинаковый масштаб для обоих, так как размеры уже нормализованы
                                    scale = float((canvas_size * 0.75) / avg_size) if avg_size > 0 else 1.0
                                    
                                    # Центр канваса
                                    center_x = canvas_size // 2
                                    center_y = canvas_size // 2
                                    
                                    # Смещения для центрирования (с небольшим смещением для видимости обеих линий)
                                    offset_x1 = float(center_x - float(centroid1[0]) * scale) - 5  # Смещение влево
                                    offset_y1 = float(center_y - float(centroid1[1]) * scale)
                                    offset_x2 = float(center_x - float(centroid2[0]) * scale) + 5  # Смещение вправо
                                    offset_y2 = float(center_y - float(centroid2[1]) * scale)
                                    
                                    # Масштабируем и смещаем точки первого элемента (зеленый)
                                    scaled_points1 = points1.astype(np.float64) * float(scale)
                                    scaled_points1[:, 0] += float(offset_x1)
                                    scaled_points1[:, 1] += float(offset_y1)
                                    
                                    # Масштабируем и смещаем точки второго элемента (красный)
                                    scaled_points2 = points2.astype(np.float64) * float(scale)
                                    scaled_points2[:, 0] += float(offset_x2)
                                    scaled_points2[:, 1] += float(offset_y2)
                                    
                                    # Определяем, замкнут ли контур (зависит от типа элемента)
                                    closed_features = ['face_oval', 'head_shape', 'left_eye', 'right_eye', 
                                                     'mouth_outer', 'mouth_inner', 'upper_lip', 'lower_lip', 
                                                     'nose_contour', 'left_ear', 'right_ear']
                                    closed = feat_name in closed_features if feat_name else True
                                    
                                    # Рисуем оба элемента на белом фоне (только линии, без изображений)
                                    vis_feat = self.visualizer.draw_feature(white_bg, scaled_points1, self.visualizer.color_green, closed)
                                    vis_feat = self.visualizer.draw_feature(vis_feat, scaled_points2, self.visualizer.color_red, closed)
                                    
                                    return vis_feat
                                except Exception as e:
                                    return None
                            
                            # Создаем наложенную визуализацию с нормализованными линиями на белом фоне
                            vis_overlay = create_overlay_visualization(points1, points2, feat_name=feat_name, target_size=500)
                            
                            if vis_overlay is None:
                                raise ValueError("Не удалось создать наложенную визуализацию")
                            
                            vis_overlay_rgb = cv2.cvtColor(vis_overlay, cv2.COLOR_BGR2RGB)
                            vis_overlay_resized = resize_for_display(vis_overlay_rgb, max_size=600)
                            vis_overlay_pil = Image.fromarray(vis_overlay_resized)
                            vis_overlay_tk = ImageTk.PhotoImage(vis_overlay_pil)
                            
                            # ЛЕВАЯ ЧАСТЬ - фото лиц с выбранной чертой (для контекста)
                            # Используем оригинальные точки для отображения на фото
                            orig_points1 = self.features1.get(feat_name, np.array([]))
                            orig_points2 = self.features2.get(feat_name, np.array([]))
                            
                            photos_label = tk.Label(left_photos_frame, text=f"{name_ru}", 
                                                   font=("Arial", 11, "bold"), bg='#f0f0f0', fg='#333')
                            photos_label.pack(pady=(0, 10))
                            
                            # Рисуем черту для первого лица на фото
                            if len(orig_points1) > 0:
                                image1_bgr = image1.copy()
                                if len(image1_bgr.shape) == 2:
                                    image1_bgr = cv2.cvtColor(image1_bgr, cv2.COLOR_GRAY2BGR)
                                # Убеждаемся, что точки имеют правильную форму
                                if not isinstance(orig_points1, np.ndarray):
                                    orig_points1 = np.array(orig_points1, dtype=np.float64)
                                if len(orig_points1.shape) != 2 or orig_points1.shape[1] != 2:
                                    orig_points1 = orig_points1.reshape(-1, 2)
                                feat_img1 = self.visualizer.draw_feature(
                                    image1_bgr.copy(), orig_points1, self.visualizer.color_green,
                                    closed=feat_name in ['face_oval', 'head_shape', 'left_eye', 
                                                       'right_eye', 'mouth_outer', 'mouth_inner',
                                                       'upper_lip', 'lower_lip', 'nose_contour']
                                )
                                feat_img1_rgb = cv2.cvtColor(feat_img1, cv2.COLOR_BGR2RGB)
                            else:
                                feat_img1_rgb = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2RGB)
                            
                            # Рисуем черту для второго лица на фото
                            if len(orig_points2) > 0:
                                image2_bgr = image2.copy()
                                if len(image2_bgr.shape) == 2:
                                    image2_bgr = cv2.cvtColor(image2_bgr, cv2.COLOR_GRAY2BGR)
                                # Убеждаемся, что точки имеют правильную форму
                                if not isinstance(orig_points2, np.ndarray):
                                    orig_points2 = np.array(orig_points2, dtype=np.float64)
                                if len(orig_points2.shape) != 2 or orig_points2.shape[1] != 2:
                                    orig_points2 = orig_points2.reshape(-1, 2)
                                feat_img2 = self.visualizer.draw_feature(
                                    image2_bgr.copy(), orig_points2, self.visualizer.color_red,
                                    closed=feat_name in ['face_oval', 'head_shape', 'left_eye', 
                                                       'right_eye', 'mouth_outer', 'mouth_inner',
                                                       'upper_lip', 'lower_lip', 'nose_contour']
                                )
                                feat_img2_rgb = cv2.cvtColor(feat_img2, cv2.COLOR_BGR2RGB)
                            else:
                                feat_img2_rgb = cv2.cvtColor(image2.copy(), cv2.COLOR_BGR2RGB)
                            
                            # Масштабируем изображения
                            feat_img1_resized = resize_for_display(feat_img1_rgb, max_size=300)
                            feat_img2_resized = resize_for_display(feat_img2_rgb, max_size=300)
                            
                            feat_img1_pil = Image.fromarray(feat_img1_resized)
                            feat_img2_pil = Image.fromarray(feat_img2_resized)
                            
                            feat_img1_tk = ImageTk.PhotoImage(feat_img1_pil)
                            feat_img2_tk = ImageTk.PhotoImage(feat_img2_pil)
                            
                            frame1 = tk.LabelFrame(left_photos_frame, text="Лицо 1 (Зеленый)", 
                                                  font=("Arial", 10, "bold"), 
                                                  bg='#f0f0f0', fg='#2E7D32', padx=10, pady=5)
                            frame1.pack(pady=10)
                            label1 = tk.Label(frame1, image=feat_img1_tk, bg='white', relief=tk.RAISED, 
                                             borderwidth=2)
                            label1.image = feat_img1_tk
                            label1.pack(padx=5, pady=5)
                            
                            frame2 = tk.LabelFrame(left_photos_frame, text="Лицо 2 (Красный)", 
                                                  font=("Arial", 10, "bold"), 
                                                  bg='#f0f0f0', fg='#C62828', padx=10, pady=5)
                            frame2.pack(pady=10)
                            label2 = tk.Label(frame2, image=feat_img2_tk, bg='white', relief=tk.RAISED, 
                                             borderwidth=2)
                            label2.image = feat_img2_tk
                            label2.pack(padx=5, pady=5)
                            
                            # ПРАВАЯ ЧАСТЬ - наложенная визуализация (как в старой версии)
                            if feat_name in self.results:
                                similarity = self.results[feat_name]
                                frame3 = tk.LabelFrame(right_overlay_frame, 
                                                      text=f"Наложение: {name_ru} (Зеленый - Лицо 1, Красный - Лицо 2) ({similarity:.1f}%)", 
                                                      font=("Arial", 11, "bold"), bg='#f0f0f0', fg='#333', 
                                                      padx=15, pady=10)
                                frame3.pack(fill=tk.BOTH, expand=True)
                                
                                overlay_inner = tk.Frame(frame3, bg='white')
                                overlay_inner.pack(expand=True, fill=tk.BOTH, padx=15, pady=15)
                                
                                label3 = tk.Label(overlay_inner, image=vis_overlay_tk, bg='white')
                                label3.image = vis_overlay_tk
                                label3.pack(expand=True)
                            
                            viz_canvas.update_idletasks()
                            viz_canvas.configure(scrollregion=viz_canvas.bbox("all"))
                        except Exception as e:
                            import traceback
                            error_msg = f"Ошибка отображения: {str(e)}\n\nДетали:\n{traceback.format_exc()}"
                            error_label = tk.Label(right_overlay_frame, 
                                                  text=error_msg, 
                                                  font=("Arial", 10), bg='white', fg='red',
                                                  justify='left', wraplength=500)
                            error_label.pack(padx=15, pady=15)
                    else:
                        # Если нет точек ни у одного лица
                        error_label = tk.Label(right_overlay_frame, 
                                              text=f"Нет данных для черты '{name_ru}' у обоих лиц", 
                                              font=("Arial", 10), bg='white', fg='orange')
                        error_label.pack(padx=15, pady=15)
            
            tk.Radiobutton(
                mode_frame, text="Общая", variable=viz_mode, value="overall",
                font=("Arial", 9), bg='#f0f0f0', command=update_viz
            ).pack(side=tk.LEFT, padx=2)
            
            tk.Radiobutton(
                mode_frame, text="По элементам", variable=viz_mode, value="detailed",
                font=("Arial", 9), bg='#f0f0f0', command=update_viz
            ).pack(side=tk.LEFT, padx=2)
            
            # Первоначальное отображение
            update_viz()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать визуализации: {e}")
            viz_window.destroy()
    
    def save_results(self):
        """Сохраняет результаты в файлы"""
        if not self.results or not self.features1 or not self.features2:
            messagebox.showwarning("Предупреждение", "Сначала выполните сравнение лиц.")
            return
        
        output_dir = filedialog.askdirectory(title="Выберите папку для сохранения результатов")
        
        if not output_dir:
            return
        
        try:
            image1 = self.features1['image']
            image2 = self.features2['image']
            
            vis1 = self.visualizer.visualize_face_features(
                image1, self.features1, self.visualizer.color_green
            )
            vis2 = self.visualizer.visualize_face_features(
                image2, self.features2, self.visualizer.color_red
            )
            overlay = self.visualizer.create_overlay_comparison(
                self.features1, self.features2, image1, image2, self.results
            )
            results_img = self.visualizer.create_results_image(self.results)
            
            cv2.imwrite(os.path.join(output_dir, "face1_annotated.jpg"), vis1)
            cv2.imwrite(os.path.join(output_dir, "face2_annotated.jpg"), vis2)
            cv2.imwrite(os.path.join(output_dir, "overlay_comparison.jpg"), overlay)
            cv2.imwrite(os.path.join(output_dir, "results.jpg"), results_img)
            
            messagebox.showinfo(
                "Успех",
                f"Результаты успешно сохранены в папку:\n{output_dir}"
            )
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить результаты: {e}")

