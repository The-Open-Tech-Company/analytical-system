"""
Модуль интерфейса для анализа одного лица
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from face_analyzer import FaceAnalyzer
from face_visualizer import FaceVisualizer
from face_database_window import FaceDatabaseWindow


class FaceAnalysisWindow:
    """Класс для создания окна анализа одного лица"""
    
    def __init__(self, parent_window, main_app):
        self.parent_window = parent_window
        self.main_app = main_app
        self.parent_window.title("Анализ лица")
        self.parent_window.geometry("1000x700")
        self.parent_window.configure(bg='#f0f0f0')
        
        # Переменные
        self.image_path = None
        self.features = None
        self.original_image = None
        
        # Инициализация компонентов
        try:
            self.analyzer = FaceAnalyzer()
            self.visualizer = FaceVisualizer()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка инициализации системы: {e}")
            return
        
        self.create_widgets()
    
    def create_widgets(self):
        """Создает виджеты интерфейса"""
        
        # Заголовок
        title_label = tk.Label(
            self.parent_window,
            text="Анализ лица",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        title_label.pack(pady=10)
        
        # Основной контейнер
        main_frame = tk.Frame(self.parent_window, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - загрузка и изображение
        left_panel = tk.Frame(main_frame, bg='#f0f0f0', width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5)
        
        # Панель загрузки
        load_frame = tk.LabelFrame(
            left_panel,
            text="Загрузка изображения",
            font=("Arial", 11, "bold"),
            bg='#f0f0f0',
            fg='#333',
            padx=10,
            pady=10
        )
        load_frame.pack(fill=tk.X, pady=5)
        
        load_btn = tk.Button(
            load_frame,
            text="📁 Загрузить фото",
            font=("Arial", 12, "bold"),
            bg='#2196F3',
            fg='white',
            padx=20,
            pady=10,
            command=self.load_image,
            cursor='hand2'
        )
        load_btn.pack(pady=10)
        
        analyze_btn = tk.Button(
            load_frame,
            text="🔍 Анализ",
            font=("Arial", 12, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            command=self.analyze_face,
            cursor='hand2'
        )
        analyze_btn.pack(pady=10)
        
        # Область для изображения
        image_frame = tk.LabelFrame(
            left_panel,
            text="Изображение",
            font=("Arial", 11, "bold"),
            bg='#f0f0f0',
            fg='#333',
            padx=10,
            pady=10
        )
        image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.image_label = tk.Label(
            image_frame,
            text="Изображение не загружено",
            bg='white',
            width=30,
            height=15,
            relief=tk.SUNKEN,
            borderwidth=1
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Правая панель - результаты
        right_panel = tk.Frame(main_frame, bg='#f0f0f0', width=500)
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
        
        # Кнопки действий
        buttons_frame = tk.Frame(right_panel, bg='#f0f0f0')
        buttons_frame.pack(pady=10)
        
        # Кнопка визуализации
        viz_btn = tk.Button(
            buttons_frame,
            text="📊 Посмотреть визуализацию",
            font=("Arial", 12, "bold"),
            bg='#FF9800',
            fg='white',
            padx=20,
            pady=10,
            command=self.show_visualization,
            cursor='hand2'
        )
        viz_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка работы с БД
        self.db_btn = tk.Button(
            buttons_frame,
            text="💾 Работа с БД",
            font=("Arial", 12, "bold"),
            bg='#9C27B0',
            fg='white',
            padx=20,
            pady=10,
            command=self.open_database_window,
            cursor='hand2',
            state=tk.DISABLED  # Изначально отключена
        )
        self.db_btn.pack(side=tk.LEFT, padx=5)
    
    def load_image(self):
        """Загружает изображение"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
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
            
            self.image_path = file_path
            self.original_image = img.copy()
            self.features = None
            
            # Отключаем кнопку работы с БД при загрузке нового изображения
            if self.db_btn:
                self.db_btn.config(state=tk.DISABLED)
            
            self.display_image(img)
            self.clear_results()
            
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror(
                "Ошибка загрузки изображения",
                f"Не удалось загрузить изображение:\n\n{error_msg}\n\n"
                f"Путь к файлу: {file_path}"
            )
    
    def display_image(self, image):
        """Отображает изображение в панели"""
        try:
            if image is None or image.size == 0:
                raise ValueError("Изображение пустое")
            
            if len(image.shape) == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((350, 350), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_pil)
            
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            
        except Exception as e:
            error_msg = f"Ошибка при отображении изображения: {e}"
            self.image_label.configure(text=error_msg, image="")
            messagebox.showerror("Ошибка отображения", error_msg)
    
    def analyze_face(self):
        """Выполняет анализ лица"""
        if self.image_path is None or self.original_image is None:
            messagebox.showwarning(
                "Предупреждение",
                "Пожалуйста, сначала загрузите изображение."
            )
            return
        
        self.clear_results()
        
        loading_label = tk.Label(
            self.results_frame,
            text="Обработка изображения...",
            font=("Arial", 12),
            bg='white'
        )
        loading_label.pack(pady=20)
        self.parent_window.update()
        
        try:
            loading_label.config(text="Извлечение характеристик лица...")
            self.parent_window.update()
            
            self.features = self.analyzer.extract_face_features(self.image_path)
            
            if self.features is None:
                messagebox.showerror(
                    "Ошибка",
                    "Лицо не найдено на изображении!\n"
                    "Убедитесь, что на изображении четко видно одно лицо (анфас)."
                )
                loading_label.destroy()
                return
            
            loading_label.destroy()
            self.display_results()
            
            # Активируем кнопку работы с БД после успешного анализа
            if self.db_btn:
                self.db_btn.config(state=tk.NORMAL)
            
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
        if not self.features:
            return
        
        title_label = tk.Label(
            self.results_frame,
            text="Результаты анализа",
            font=("Arial", 14, "bold"),
            bg='white',
            fg='#333'
        )
        title_label.pack(pady=10)
        
        # Информация о поле, возрасте и расе
        info_frame = tk.Frame(self.results_frame, bg='white')
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        gender = self.features.get('gender', 'Не определен')
        age = self.features.get('age', 'Не определен')
        race = self.features.get('race', 'Не определена')
        
        gender_conf = self.features.get('gender_confidence', 0.0)
        age_conf = self.features.get('age_confidence', 0.0)
        race_conf = self.features.get('race_confidence', 0.0)
        
        # Убеждаемся, что уверенность в диапазоне [0, 1] и положительная
        gender_conf = max(0.0, min(1.0, abs(float(gender_conf))))
        age_conf = max(0.0, min(1.0, abs(float(age_conf))))
        race_conf = max(0.0, min(1.0, abs(float(race_conf))))
        
        info_text = f"Пол: {gender} (уверенность: {gender_conf*100:.0f}%)\n"
        info_text += f"Возраст: {age} (уверенность: {age_conf*100:.0f}%)\n"
        info_text += f"Раса: {race} (уверенность: {race_conf*100:.0f}%)"
        
        info_label = tk.Label(
            info_frame,
            text=info_text,
            font=("Arial", 11),
            bg='white',
            fg='#333',
            justify='left'
        )
        info_label.pack(anchor='w', pady=5)
        
        # Разделитель
        separator = tk.Frame(self.results_frame, height=2, bg='#ccc')
        separator.pack(fill=tk.X, padx=10, pady=10)
        
        # Список проанализированных черт
        features_title = tk.Label(
            self.results_frame,
            text="Проанализированные черты лица",
            font=("Arial", 12, "bold"),
            bg='white',
            fg='#333'
        )
        features_title.pack(pady=5)
        
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
            'forehead': 'Лоб'
        }
        
        # Отображаем список черт с количеством точек
        for feature_name, name_ru in feature_names_ru.items():
            points = self.features.get(feature_name, np.array([]))
            if len(points) > 0:
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
                
                count_label = tk.Label(
                    feature_frame,
                    text=f"Точек: {len(points)}",
                    font=("Arial", 10),
                    bg='white',
                    fg='#666'
                )
                count_label.pack(side=tk.LEFT, padx=5)
        
        # Обновляем прокрутку
        self.results_canvas.update_idletasks()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        self.results_canvas.yview_moveto(0)
    
    def show_visualization(self):
        """Открывает окно с визуализациями"""
        if not self.features:
            messagebox.showwarning("Предупреждение", "Сначала выполните анализ лица.")
            return
        
        # Создаем новое окно для визуализаций
        viz_window = tk.Toplevel(self.parent_window)
        viz_window.transient(self.parent_window)  # Не будет перекрывать родительское окно
        viz_window.title("Визуализация анализа лица")
        viz_window.geometry("1600x1000")
        viz_window.configure(bg='#f0f0f0')
        
        # Переменные для режима отображения
        viz_mode = tk.StringVar(value="overall")
        background_mode = tk.StringVar(value="photo")
        selected_feature = tk.StringVar(value="")
        
        # Переменные для яркости и контрастности
        brightness_var = tk.DoubleVar(value=0.0)  # -100 до 100
        contrast_var = tk.DoubleVar(value=1.0)    # 0.5 до 2.0
        zoom_var = tk.DoubleVar(value=1.0)        # 1.0 до 5.0 (коэффициент приближения)
        sharpness_var = tk.DoubleVar(value=0.0)    # 0.0 до 2.0 (коэффициент резкости)
        points_density_var = tk.IntVar(value=50)  # 10 до 200 (количество точек на черту)
        
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
            'forehead': 'Лоб'
        }
        
        # Получаем список доступных черт
        available_features = []
        for feat_name in feature_names_ru.keys():
            points = self.features.get(feat_name, np.array([]))
            if len(points) > 0:
                available_features.append(feat_name)
        
        try:
            image = self.features['image'].copy()
            
            # Убеждаемся, что изображение в правильном формате
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # Создаем визуализацию
            vis = self.visualizer.visualize_face_features(
                image, self.features, self.visualizer.color_green
            )
            
            # Конвертируем в формат для tkinter (RGB)
            if len(vis.shape) == 3 and vis.shape[2] == 3:
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            else:
                vis_rgb = vis.copy()
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image.copy()
            
            # Масштабируем изображения
            def resize_for_display(img, max_size=900):
                h, w = img.shape[:2]
                scale = min(max_size / w, max_size / h, 1.0)
                new_w = int(w * scale)
                new_h = int(h * scale)
                return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Функция для применения яркости и контрастности
            def apply_brightness_contrast(img, brightness=0, contrast=1.0):
                """Применяет яркость и контрастность к изображению"""
                img = img.astype(np.float32)
                
                # Применяем контрастность
                img = img * contrast
                
                # Применяем яркость (brightness в диапазоне -100 до 100, конвертируем в -255 до 255)
                brightness_adj = brightness * 2.55
                img = img + brightness_adj
                
                # Ограничиваем значения в диапазоне [0, 255]
                img = np.clip(img, 0, 255)
                
                return img.astype(np.uint8)
            
            # Функция для повышения резкости
            def apply_sharpness(img, sharpness=0.0):
                """Применяет повышение резкости к изображению"""
                if sharpness <= 0.0:
                    return img
                
                # Создаем ядро для повышения резкости (unsharp mask)
                # Чем больше sharpness, тем сильнее эффект
                kernel = np.array([
                    [0, -sharpness, 0],
                    [-sharpness, 1 + 4 * sharpness, -sharpness],
                    [0, -sharpness, 0]
                ])
                
                # Применяем свертку
                sharpened = cv2.filter2D(img, -1, kernel)
                
                # Ограничиваем значения в диапазоне [0, 255]
                sharpened = np.clip(sharpened, 0, 255)
                
                return sharpened.astype(np.uint8)
            
            # Функция для обрезки и увеличения области вокруг черты
            def crop_and_zoom_feature(image, points, zoom_factor=1.0, padding_ratio=0.3):
                """Обрезает и увеличивает область вокруг черты лица"""
                if len(points) == 0:
                    return image, points
                
                # Вычисляем границы области черты
                min_x = int(np.min(points[:, 0]))
                max_x = int(np.max(points[:, 0]))
                min_y = int(np.min(points[:, 1]))
                max_y = int(np.max(points[:, 1]))
                
                # Добавляем отступы
                width = max_x - min_x
                height = max_y - min_y
                padding_x = int(width * padding_ratio)
                padding_y = int(height * padding_ratio)
                
                # Получаем размеры изображения
                img_h, img_w = image.shape[:2]
                
                # Вычисляем координаты обрезки с учетом границ изображения
                crop_x1 = max(0, min_x - padding_x)
                crop_y1 = max(0, min_y - padding_y)
                crop_x2 = min(img_w, max_x + padding_x)
                crop_y2 = min(img_h, max_y + padding_y)
                
                # Обрезаем изображение
                cropped_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Применяем приближение
                if zoom_factor > 1.0:
                    new_w = int(cropped_img.shape[1] * zoom_factor)
                    new_h = int(cropped_img.shape[0] * zoom_factor)
                    cropped_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Обновляем координаты точек относительно обрезанного изображения
                adjusted_points = points.copy().astype(np.float32)
                adjusted_points[:, 0] -= crop_x1
                adjusted_points[:, 1] -= crop_y1
                
                # Масштабируем точки, если применено приближение
                if zoom_factor > 1.0:
                    adjusted_points *= zoom_factor
                
                return cropped_img, adjusted_points
            
            # Создаем словарь для отображения ПЕРЕД созданием виджетов
            feature_display_map = {feature_names_ru.get(f, f): f for f in available_features}
            
            # Основной горизонтальный контейнер
            main_container = tk.Frame(viz_window, bg='#f0f0f0')
            main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Левая панель управления (упаковываем ПЕРВОЙ)
            left_panel = tk.Frame(main_container, bg='#f0f0f0', width=300)
            left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
            left_panel.pack_propagate(False)
            left_panel.config(width=300)
            
            # Панель управления
            control_frame = tk.LabelFrame(
                left_panel,
                text="Управление",
                font=("Arial", 10, "bold"),
                bg='#f0f0f0',
                padx=8,
                pady=6
            )
            control_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Основной контейнер с прокруткой (справа, упаковываем ВТОРЫМ)
            main_viz_container = tk.Frame(main_container, bg='#f0f0f0')
            main_viz_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            canvas = tk.Canvas(main_viz_container, bg='#f0f0f0', highlightthickness=0)
            scrollbar = ttk.Scrollbar(main_viz_container, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            # Создаем окно в canvas и сохраняем его ID
            canvas_window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            def center_canvas_content():
                """Центрирует содержимое canvas по горизонтали"""
                canvas.update_idletasks()
                canvas_width = canvas.winfo_width()
                scrollable_width = scrollable_frame.winfo_reqwidth()
                if canvas_width > scrollable_width and canvas_width > 1:
                    x = (canvas_width - scrollable_width) // 2
                    canvas.coords(canvas_window_id, x, 0)
            
            # Привязываем центрирование к изменению размера
            def on_canvas_configure(event):
                center_canvas_content()
            
            canvas.bind('<Configure>', on_canvas_configure)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Переключатель режима визуализации
            mode_frame = tk.Frame(control_frame, bg='#f0f0f0')
            mode_frame.pack(fill=tk.X, pady=5, padx=5)
            
            tk.Label(mode_frame, text="Режим:", font=("Arial", 9, "bold"), bg='#f0f0f0').pack(anchor='w', pady=2)
            mode_buttons = tk.Frame(mode_frame, bg='#f0f0f0')
            mode_buttons.pack(fill=tk.X)
            
            # Переключатель фона
            bg_frame = tk.Frame(control_frame, bg='#f0f0f0')
            bg_frame.pack(fill=tk.X, pady=5, padx=5)
            
            tk.Label(bg_frame, text="Фон:", font=("Arial", 9, "bold"), bg='#f0f0f0').pack(anchor='w', pady=2)
            bg_buttons = tk.Frame(bg_frame, bg='#f0f0f0')
            bg_buttons.pack(fill=tk.X)
            
            # Панель настроек изображения (яркость, контрастность, приближение)
            image_settings_frame = tk.Frame(control_frame, bg='#f0f0f0')
            
            # Яркость
            brightness_row = tk.Frame(image_settings_frame, bg='#f0f0f0')
            brightness_row.pack(fill=tk.X, pady=3)
            tk.Label(brightness_row, text="Яркость:", font=("Arial", 9, "bold"), bg='#f0f0f0').pack(anchor='w', pady=2)
            brightness_scale_frame = tk.Frame(brightness_row, bg='#f0f0f0')
            brightness_scale_frame.pack(fill=tk.X)
            brightness_scale = tk.Scale(brightness_scale_frame, from_=-100, to=100, orient=tk.HORIZONTAL, 
                                       variable=brightness_var, length=200)
            brightness_scale.pack(side=tk.LEFT, padx=5)
            brightness_value_label = tk.Label(brightness_scale_frame, text="0", font=("Arial", 8), bg='#f0f0f0', width=5)
            brightness_value_label.pack(side=tk.LEFT, padx=2)
            
            # Контрастность
            contrast_row = tk.Frame(image_settings_frame, bg='#f0f0f0')
            contrast_row.pack(fill=tk.X, pady=3)
            tk.Label(contrast_row, text="Контраст:", font=("Arial", 9, "bold"), bg='#f0f0f0').pack(anchor='w', pady=2)
            contrast_scale_frame = tk.Frame(contrast_row, bg='#f0f0f0')
            contrast_scale_frame.pack(fill=tk.X)
            contrast_scale = tk.Scale(contrast_scale_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
                                     variable=contrast_var, length=200)
            contrast_scale.pack(side=tk.LEFT, padx=5)
            contrast_value_label = tk.Label(contrast_scale_frame, text="1.0", font=("Arial", 8), bg='#f0f0f0', width=5)
            contrast_value_label.pack(side=tk.LEFT, padx=2)
            
            # Приближение
            zoom_row = tk.Frame(image_settings_frame, bg='#f0f0f0')
            zoom_row.pack(fill=tk.X, pady=3)
            tk.Label(zoom_row, text="Приближение:", font=("Arial", 9, "bold"), bg='#f0f0f0').pack(anchor='w', pady=2)
            zoom_scale_frame = tk.Frame(zoom_row, bg='#f0f0f0')
            zoom_scale_frame.pack(fill=tk.X)
            zoom_scale = tk.Scale(zoom_scale_frame, from_=1.0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL,
                                 variable=zoom_var, length=200)
            zoom_scale.pack(side=tk.LEFT, padx=5)
            zoom_value_label = tk.Label(zoom_scale_frame, text="1.0x", font=("Arial", 8), bg='#f0f0f0', width=5)
            zoom_value_label.pack(side=tk.LEFT, padx=2)
            
            # Резкость
            sharpness_row = tk.Frame(image_settings_frame, bg='#f0f0f0')
            sharpness_row.pack(fill=tk.X, pady=3)
            tk.Label(sharpness_row, text="Резкость:", font=("Arial", 9, "bold"), bg='#f0f0f0').pack(anchor='w', pady=2)
            sharpness_scale_frame = tk.Frame(sharpness_row, bg='#f0f0f0')
            sharpness_scale_frame.pack(fill=tk.X)
            sharpness_scale = tk.Scale(sharpness_scale_frame, from_=0.0, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
                                      variable=sharpness_var, length=200)
            sharpness_scale.pack(side=tk.LEFT, padx=5)
            sharpness_value_label = tk.Label(sharpness_scale_frame, text="0.0", font=("Arial", 8), bg='#f0f0f0', width=5)
            sharpness_value_label.pack(side=tk.LEFT, padx=2)
            
            # Плотность точек (количество точек на черту)
            points_density_row = tk.Frame(image_settings_frame, bg='#f0f0f0')
            points_density_row.pack(fill=tk.X, pady=3)
            tk.Label(points_density_row, text="Точек на черту:", font=("Arial", 9, "bold"), bg='#f0f0f0').pack(anchor='w', pady=2)
            points_density_scale_frame = tk.Frame(points_density_row, bg='#f0f0f0')
            points_density_scale_frame.pack(fill=tk.X)
            points_density_scale = tk.Scale(points_density_scale_frame, from_=10, to=200, resolution=5, orient=tk.HORIZONTAL,
                                          variable=points_density_var, length=200)
            points_density_scale.pack(side=tk.LEFT, padx=5)
            points_density_value_label = tk.Label(points_density_scale_frame, text="50", font=("Arial", 8), bg='#f0f0f0', width=5)
            points_density_value_label.pack(side=tk.LEFT, padx=2)
            
            # Изначально скрываем настройки изображения
            image_settings_frame.pack_forget()
            
            # Выбор черты (только для режима "По элементам")
            feature_select_frame = tk.Frame(control_frame, bg='#f0f0f0')
            # Изначально скрыт, будет показан только в режиме "По элементам"
            feature_select_frame.pack_forget()
            
            tk.Label(feature_select_frame, text="Выберите черту лица:", font=("Arial", 9, "bold"), bg='#f0f0f0').pack(anchor='w', pady=2, padx=5)
            
            feature_combo = ttk.Combobox(
                feature_select_frame,
                textvariable=selected_feature,
                values=list(feature_display_map.keys()) if feature_display_map else [],
                state="readonly",
                width=22,
                font=("Arial", 9)
            )
            feature_combo.pack(fill=tk.X, padx=5, pady=2)
            
            if available_features and feature_display_map:
                first_display = list(feature_display_map.keys())[0]
                feature_combo.current(0)
                selected_feature.set(first_display)
            
            def update_viz():
                # Показываем/скрываем панель настроек изображения
                # image_settings_frame уже является дочерним элементом control_frame (который в left_panel)
                if background_mode.get() == "photo":
                    # Сначала забываем, если уже упакована
                    try:
                        image_settings_frame.pack_forget()
                    except:
                        pass
                    # Затем упаковываем в control_frame (который находится в left_panel слева)
                    image_settings_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                else:
                    image_settings_frame.pack_forget()
                
                # Показываем/скрываем выбор черты
                if viz_mode.get() == "detailed":
                    if available_features and feature_display_map:
                        feature_select_frame.pack(fill=tk.X, padx=5, pady=5, before=image_settings_frame if background_mode.get() == "photo" else None)
                    else:
                        feature_select_frame.pack_forget()
                        # Показываем сообщение об отсутствии черт
                        for widget in scrollable_frame.winfo_children():
                            widget.destroy()
                        error_frame = tk.LabelFrame(
                            scrollable_frame,
                            text="Ошибка",
                            font=("Arial", 12, "bold"),
                            bg='#f0f0f0',
                            padx=10,
                            pady=10
                        )
                        error_frame.pack(pady=20, padx=20)
                        error_label = tk.Label(
                            error_frame,
                            text="Нет доступных черт лица для отображения",
                            font=("Arial", 10),
                            bg='white',
                            fg='red'
                        )
                        error_label.pack()
                        return
                else:
                    feature_select_frame.pack_forget()
                
                for widget in scrollable_frame.winfo_children():
                    widget.destroy()
                
                if viz_mode.get() == "overall":
                    # Общая визуализация
                    # Получаем количество точек для визуализации
                    max_points = int(points_density_var.get())
                    
                    if background_mode.get() == "photo":
                        # Создаем визуализацию с ограничением количества точек
                        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) if len(image_rgb.shape) == 3 else image_rgb
                        vis_img = self.visualizer.visualize_face_features(
                            image_bgr.copy(), self.features, self.visualizer.color_green, 
                            max_points_per_feature=max_points
                        )
                        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                        
                        # Применяем яркость и контрастность
                        vis_img = apply_brightness_contrast(
                            vis_img, 
                            brightness_var.get(), 
                            contrast_var.get()
                        )
                        # Применяем резкость
                        vis_img = apply_sharpness(vis_img, sharpness_var.get())
                        
                        # Применяем приближение
                        zoom_factor = zoom_var.get()
                        if zoom_factor > 1.0:
                            h, w = vis_img.shape[:2]
                            new_w = int(w * zoom_factor)
                            new_h = int(h * zoom_factor)
                            vis_img = cv2.resize(vis_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        # Белый фон
                        h, w = vis_rgb.shape[:2]
                        white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
                        # Рисуем только линии на белом фоне с ограничением количества точек
                        vis_img = self.visualizer.visualize_face_features(
                            white_bg, self.features, self.visualizer.color_green,
                            max_points_per_feature=max_points
                        )
                        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                        
                        # Применяем приближение для белого фона тоже
                        zoom_factor = zoom_var.get()
                        if zoom_factor > 1.0:
                            h, w = vis_img.shape[:2]
                            new_w = int(w * zoom_factor)
                            new_h = int(h * zoom_factor)
                            vis_img = cv2.resize(vis_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Увеличиваем базовый размер для общей визуализации
                    vis_resized = resize_for_display(vis_img, max_size=1400)
                    vis_pil = Image.fromarray(vis_resized)
                    vis_tk = ImageTk.PhotoImage(vis_pil)
                    
                    frame = tk.LabelFrame(
                        scrollable_frame,
                        text="Общая визуализация всех черт лица",
                        font=("Arial", 11, "bold"),
                        bg='#f0f0f0',
                        padx=10,
                        pady=10
                    )
                    frame.pack(pady=10, fill=tk.BOTH, expand=True)
                    
                    # Центрируем изображение
                    image_container = tk.Frame(frame, bg='white')
                    image_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    label = tk.Label(image_container, image=vis_tk, bg='white')
                    label.image = vis_tk
                    label.pack(anchor='center')
                    
                else:
                    # Поэлементная визуализация - показываем только выбранную черту
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
                    
                    points = self.features.get(feat_name, np.array([]))
                    if len(points) > 0:
                        name_ru = feature_names_ru.get(feat_name, feat_name)
                        
                        try:
                            # Убеждаемся, что points - это numpy array с правильной формой
                            if not isinstance(points, np.ndarray):
                                points = np.array(points, dtype=np.float64)
                            
                            if len(points.shape) != 2 or points.shape[1] != 2:
                                # Пытаемся исправить форму
                                points = points.reshape(-1, 2)
                            
                            # Убеждаемся, что координаты валидны
                            points = points.astype(np.float64)
                            
                            def create_single_feature_visualization(points, feat_name, target_size=500):
                                """Создает визуализацию одного элемента на белом фоне с нормализацией (как в старой версии)"""
                                if len(points) == 0:
                                    return None
                                
                                try:
                                    # Вычисляем центроид для центрирования
                                    centroid = np.mean(points, axis=0)
                                    centroid = np.array([float(centroid[0]), float(centroid[1])])
                                    
                                    # Вычисляем размер элемента
                                    size = np.max(points, axis=0) - np.min(points, axis=0)
                                    avg_size = float(np.mean(size))
                                    
                                    # Определяем размер канваса
                                    canvas_size = target_size
                                    
                                    # Создаем белый фон
                                    white_bg = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
                                    
                                    # Вычисляем масштаб для нормализации
                                    scale = float((canvas_size * 0.75) / avg_size) if avg_size > 0 else 1.0
                                    
                                    # Центр канваса
                                    center_x = canvas_size // 2
                                    center_y = canvas_size // 2
                                    
                                    # Смещения для центрирования
                                    offset_x = float(center_x - float(centroid[0]) * scale)
                                    offset_y = float(center_y - float(centroid[1]) * scale)
                                    
                                    # Масштабируем и смещаем точки
                                    scaled_points = points.astype(np.float64) * float(scale)
                                    scaled_points[:, 0] += float(offset_x)
                                    scaled_points[:, 1] += float(offset_y)
                                    
                                    # Определяем, замкнут ли контур
                                    closed_features = ['face_oval', 'head_shape', 'left_eye', 'right_eye', 
                                                     'mouth_outer', 'mouth_inner', 'upper_lip', 'lower_lip', 
                                                     'nose_contour', 'left_ear', 'right_ear']
                                    closed = feat_name in closed_features
                                    
                                    # Рисуем элемент на белом фоне
                                    vis_feat = self.visualizer.draw_feature(white_bg, scaled_points, self.visualizer.color_green, closed)
                                    
                                    return vis_feat
                                except Exception as e:
                                    return None
                            
                            if background_mode.get() == "photo":
                                # На фото - обрезаем и увеличиваем область вокруг черты
                                zoom_factor = zoom_var.get()
                                
                                # Конвертируем в BGR для обработки
                                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) if len(image_rgb.shape) == 3 else image_rgb
                                
                                # Обрезаем и увеличиваем область (увеличиваем padding для более крупного обрезания)
                                cropped_img, adjusted_points = crop_and_zoom_feature(
                                    image_bgr, points, zoom_factor=zoom_factor, padding_ratio=0.5
                                )
                                
                                # Применяем яркость и контрастность к обрезанному изображению
                                cropped_img = apply_brightness_contrast(
                                    cropped_img,
                                    brightness_var.get(),
                                    contrast_var.get()
                                )
                                
                                # Применяем резкость
                                cropped_img = apply_sharpness(cropped_img, sharpness_var.get())
                                
                                # Рисуем черту на обрезанном изображении
                                feat_img = self.visualizer.draw_feature(
                                    cropped_img.copy(), adjusted_points, self.visualizer.color_green,
                                    closed=feat_name in ['face_oval', 'head_shape', 'left_eye', 
                                                       'right_eye', 'mouth_outer', 'mouth_inner',
                                                       'upper_lip', 'lower_lip', 'nose_contour']
                                )
                                
                                # Конвертируем обратно в RGB для отображения
                                if len(feat_img.shape) == 3 and feat_img.shape[2] == 3:
                                    feat_img = cv2.cvtColor(feat_img, cv2.COLOR_BGR2RGB)
                            else:
                                # Белый фон - используем метод из старой версии с нормализацией
                                feat_img = create_single_feature_visualization(points, feat_name, target_size=800)
                                if feat_img is None:
                                    raise ValueError("Не удалось создать визуализацию")
                                # Конвертируем в RGB если нужно
                                if len(feat_img.shape) == 3 and feat_img.shape[2] == 3:
                                    feat_img = cv2.cvtColor(feat_img, cv2.COLOR_BGR2RGB)
                                
                                # Применяем резкость к белому фону тоже
                                feat_img = apply_sharpness(feat_img, sharpness_var.get())
                            
                            # Увеличиваем базовый размер для детальной визуализации
                            feat_resized = resize_for_display(feat_img, max_size=1200)
                            feat_pil = Image.fromarray(feat_resized)
                            feat_tk = ImageTk.PhotoImage(feat_pil)
                            
                            frame = tk.LabelFrame(
                                scrollable_frame,
                                text=f"{name_ru} ({len(points)} точек)",
                                font=("Arial", 12, "bold"),
                                bg='#f0f0f0',
                                padx=10,
                                pady=10
                            )
                            frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
                            
                            # Центрируем изображение
                            image_container = tk.Frame(frame, bg='white')
                            image_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                            
                            label = tk.Label(image_container, image=feat_tk, bg='white')
                            label.image = feat_tk
                            label.pack(anchor='center')
                        except Exception as e:
                            # Если ошибка, показываем сообщение
                            error_frame = tk.LabelFrame(
                                scrollable_frame,
                                text=f"{name_ru} - Ошибка отображения",
                                font=("Arial", 12, "bold"),
                                bg='#f0f0f0',
                                padx=10,
                                pady=10
                            )
                            error_frame.pack(pady=20, padx=20)
                            error_label = tk.Label(
                                error_frame,
                                text=f"Не удалось отобразить: {str(e)}",
                                font=("Arial", 10),
                                bg='white',
                                fg='red'
                            )
                            error_label.pack()
                
                canvas.update_idletasks()
                canvas.configure(scrollregion=canvas.bbox("all"))
                canvas.yview_moveto(0)
                
                # Центрируем содержимое по горизонтали
                canvas_width = canvas.winfo_width()
                scrollable_width = scrollable_frame.winfo_reqwidth()
                if canvas_width > scrollable_width and canvas_width > 1:
                    x = (canvas_width - scrollable_width) // 2
                    canvas.coords(canvas_window_id, x, 0)
            
            # Настраиваем обработчики для слайдеров
            def update_brightness_label(val):
                brightness_value_label.config(text=f"{int(float(val))}")
                update_viz()
            
            def update_contrast_label(val):
                contrast_value_label.config(text=f"{float(val):.1f}")
                update_viz()
            
            def update_zoom_label(val):
                zoom_value_label.config(text=f"{float(val):.1f}x")
                update_viz()
            
            def update_sharpness_label(val):
                sharpness_value_label.config(text=f"{float(val):.1f}")
                update_viz()
            
            def update_points_density_label(val):
                points_density_value_label.config(text=f"{int(float(val))}")
                update_viz()
            
            brightness_scale.config(command=update_brightness_label)
            contrast_scale.config(command=update_contrast_label)
            zoom_scale.config(command=update_zoom_label)
            sharpness_scale.config(command=update_sharpness_label)
            points_density_scale.config(command=update_points_density_label)
            
            # Создаем радиокнопки ПОСЛЕ определения update_viz
            tk.Radiobutton(
                mode_buttons, text="Общий", variable=viz_mode, value="overall",
                font=("Arial", 9), bg='#f0f0f0', command=update_viz
            ).pack(anchor='w', padx=5)
            
            tk.Radiobutton(
                mode_buttons, text="По элементам", variable=viz_mode, value="detailed",
                font=("Arial", 9), bg='#f0f0f0', command=update_viz
            ).pack(anchor='w', padx=5)
            
            tk.Radiobutton(
                bg_buttons, text="На фото", variable=background_mode, value="photo",
                font=("Arial", 9), bg='#f0f0f0', command=update_viz
            ).pack(anchor='w', padx=5)
            
            tk.Radiobutton(
                bg_buttons, text="Белый фон", variable=background_mode, value="white",
                font=("Arial", 9), bg='#f0f0f0', command=update_viz
            ).pack(anchor='w', padx=5)
            
            feature_combo.bind("<<ComboboxSelected>>", lambda e: update_viz())
            
            # Первоначальное отображение
            update_viz()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать визуализации: {e}")
            viz_window.destroy()
    
    def open_database_window(self):
        """Открывает окно работы с базой данных"""
        if not self.features:
            messagebox.showwarning("Предупреждение", "Сначала выполните анализ лица.")
            return
        
        try:
            window = tk.Toplevel(self.parent_window)
            window.transient(self.parent_window)
            
            app = FaceDatabaseWindow(window, self.main_app, self.features)
            self.main_app.open_windows.append(window)
            
            # Обработчик закрытия окна
            def on_close():
                if window in self.main_app.open_windows:
                    self.main_app.open_windows.remove(window)
                window.destroy()
            
            window.protocol("WM_DELETE_WINDOW", on_close)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть окно работы с БД: {e}")

