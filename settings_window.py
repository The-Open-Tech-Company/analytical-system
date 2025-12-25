"""
Модуль интерфейса для настроек и проверки элементов программы
"""
import tkinter as tk
from tkinter import ttk, messagebox
import sys


class SettingsWindow:
    """Класс для создания окна настроек"""
    
    def __init__(self, parent_window, main_app):
        self.parent_window = parent_window
        self.main_app = main_app
        self.parent_window.title("Настройки")
        self.parent_window.geometry("800x600")
        self.parent_window.configure(bg='#f0f0f0')
        
        # Словарь состояний компонентов
        self.components_status = {}
        self.components_enabled = {}
        
        self.create_widgets()
        self.check_components()
    
    def create_widgets(self):
        """Создает виджеты интерфейса"""
        
        # Заголовок
        title_label = tk.Label(
            self.parent_window,
            text="Настройки системы",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        title_label.pack(pady=10)
        
        # Основной контейнер
        main_frame = tk.Frame(self.parent_window, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Область для проверки компонентов
        check_frame = tk.LabelFrame(
            main_frame,
            text="Проверка компонентов системы",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#333',
            padx=10,
            pady=10
        )
        check_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Canvas для прокрутки
        canvas = tk.Canvas(check_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(check_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.check_canvas = canvas
        self.check_frame = scrollable_frame
        
        # Кнопка перепроверки
        refresh_btn = tk.Button(
            main_frame,
            text="🔄 Перепроверить компоненты",
            font=("Arial", 11, "bold"),
            bg='#2196F3',
            fg='white',
            padx=20,
            pady=10,
            command=self.check_components,
            cursor='hand2'
        )
        refresh_btn.pack(pady=10)
    
    def check_components(self):
        """Проверяет корректность работы всех компонентов"""
        # Очищаем предыдущие результаты
        for widget in self.check_frame.winfo_children():
            widget.destroy()
        
        # Список компонентов для проверки
        components = [
            {
                'name': 'FaceAnalyzer',
                'module': 'face_analyzer',
                'class': 'FaceAnalyzer',
                'description': 'Анализатор лиц (MediaPipe)'
            },
            {
                'name': 'FaceComparator',
                'module': 'face_comparator',
                'class': 'FaceComparator',
                'description': 'Сравниватель лиц'
            },
            {
                'name': 'FaceVisualizer',
                'module': 'face_visualizer',
                'class': 'FaceVisualizer',
                'description': 'Визуализатор результатов'
            },
            {
                'name': 'GenderAgeDNN',
                'module': 'gender_age_dnn',
                'class': 'GenderAgeDNN',
                'description': 'Определение пола и возраста (DNN)',
                'optional': True
            },
            {
                'name': 'OpenCV',
                'module': 'cv2',
                'class': None,
                'description': 'Библиотека компьютерного зрения'
            },
            {
                'name': 'NumPy',
                'module': 'numpy',
                'class': None,
                'description': 'Библиотека для работы с массивами'
            },
            {
                'name': 'PIL/Pillow',
                'module': 'PIL',
                'class': None,
                'description': 'Библиотека для работы с изображениями'
            },
            {
                'name': 'MediaPipe',
                'module': 'mediapipe',
                'class': None,
                'description': 'Библиотека для распознавания лиц'
            }
        ]
        
        # Проверяем каждый компонент
        for component in components:
            self.check_component(component)
        
        # Обновляем прокрутку
        self.check_canvas.update_idletasks()
        self.check_canvas.configure(scrollregion=self.check_canvas.bbox("all"))
        self.check_canvas.yview_moveto(0)
    
    def check_component(self, component):
        """Проверяет один компонент"""
        name = component['name']
        module_name = component['module']
        class_name = component.get('class')
        description = component.get('description', name)
        is_optional = component.get('optional', False)
        
        # Создаем фрейм для компонента
        comp_frame = tk.Frame(self.check_frame, bg='white', relief=tk.RAISED, borderwidth=1)
        comp_frame.pack(fill=tk.X, padx=5, pady=3)
        
        # Название компонента
        name_frame = tk.Frame(comp_frame, bg='white')
        name_frame.pack(fill=tk.X, padx=10, pady=5)
        
        name_label = tk.Label(
            name_frame,
            text=name,
            font=("Arial", 11, "bold"),
            bg='white',
            fg='#333',
            anchor='w'
        )
        name_label.pack(side=tk.LEFT)
        
        # Описание
        desc_label = tk.Label(
            name_frame,
            text=f"({description})",
            font=("Arial", 9),
            bg='white',
            fg='#666'
        )
        desc_label.pack(side=tk.LEFT, padx=5)
        
        # Статус
        status_frame = tk.Frame(comp_frame, bg='white')
        status_frame.pack(fill=tk.X, padx=10, pady=2)
        
        try:
            # Пытаемся импортировать модуль
            module = __import__(module_name)
            
            if class_name:
                # Проверяем класс
                if hasattr(module, class_name):
                    # Пытаемся создать экземпляр
                    try:
                        cls = getattr(module, class_name)
                        if class_name == 'FaceAnalyzer':
                            instance = cls()
                        elif class_name == 'FaceComparator':
                            instance = cls()
                        elif class_name == 'FaceVisualizer':
                            instance = cls()
                        elif class_name == 'GenderAgeDNN':
                            instance = cls()
                            if not instance.is_available():
                                raise Exception("Модели не загружены")
                        else:
                            instance = cls()
                        
                        status_text = "✓ Работает корректно"
                        status_color = '#4CAF50'
                        status = 'ok'
                        
                    except Exception as e:
                        if is_optional:
                            status_text = f"⚠ Опциональный компонент: {str(e)[:50]}"
                            status_color = '#FF9800'
                            status = 'optional'
                        else:
                            status_text = f"✗ Ошибка инициализации: {str(e)[:50]}"
                            status_color = '#F44336'
                            status = 'error'
                else:
                    status_text = f"✗ Класс {class_name} не найден"
                    status_color = '#F44336'
                    status = 'error'
            else:
                # Просто проверяем модуль
                status_text = "✓ Модуль доступен"
                status_color = '#4CAF50'
                status = 'ok'
                
        except ImportError as e:
            if is_optional:
                status_text = "⚠ Опциональный компонент недоступен"
                status_color = '#FF9800'
                status = 'optional'
            else:
                status_text = f"✗ Модуль не найден: {module_name}"
                status_color = '#F44336'
                status = 'error'
        except Exception as e:
            if is_optional:
                status_text = f"⚠ Опциональный компонент: {str(e)[:50]}"
                status_color = '#FF9800'
                status = 'optional'
            else:
                status_text = f"✗ Ошибка: {str(e)[:50]}"
                status_color = '#F44336'
                status = 'error'
        
        status_label = tk.Label(
            status_frame,
            text=status_text,
            font=("Arial", 9),
            bg='white',
            fg=status_color,
            anchor='w'
        )
        status_label.pack(side=tk.LEFT)
        
        # Чекбокс для включения/отключения (только для основных компонентов)
        if not is_optional and status == 'ok':
            enabled = self.components_enabled.get(name, True)
            enabled_var = tk.BooleanVar(value=enabled)
            
            def toggle_component(comp_name, var):
                self.components_enabled[comp_name] = var.get()
            
            checkbox = tk.Checkbutton(
                status_frame,
                text="Включен",
                variable=enabled_var,
                command=lambda: toggle_component(name, enabled_var),
                bg='white',
                font=("Arial", 9)
            )
            checkbox.pack(side=tk.RIGHT, padx=5)
            
            self.components_status[name] = {
                'status': status,
                'enabled': enabled_var
            }
        else:
            self.components_status[name] = {
                'status': status,
                'enabled': None
            }
        
        # Сохраняем информацию о компоненте
        self.components_status[name]['info'] = {
            'module': module_name,
            'class': class_name,
            'description': description,
            'optional': is_optional
        }


