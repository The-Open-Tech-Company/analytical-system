"""
Окно для работы с базой данных лиц
"""
import tkinter as tk
from tkinter import ttk, messagebox
from face_database import FaceDatabase
from face_comparator import FaceComparator
from typing import Dict, Optional


class FaceDatabaseWindow:
    """Класс для создания окна работы с базой данных лиц"""
    
    def __init__(self, parent_window, main_app, face_features: Optional[Dict] = None):
        """
        Инициализирует окно работы с БД
        
        Args:
            parent_window: Родительское окно
            main_app: Главное приложение
            face_features: Характеристики лица для поиска/добавления (опционально)
        """
        self.parent_window = parent_window
        self.main_app = main_app
        self.face_features = face_features
        
        self.parent_window.title("Работа с базой данных")
        self.parent_window.geometry("900x700")
        self.parent_window.configure(bg='#f0f0f0')
        
        # Инициализируем базу данных
        self.db = FaceDatabase()
        
        self.create_widgets()
    
    def create_widgets(self):
        """Создает виджеты интерфейса"""
        
        # Заголовок
        title_label = tk.Label(
            self.parent_window,
            text="Работа с базой данных лиц",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        title_label.pack(pady=10)
        
        # Основной контейнер
        main_frame = tk.Frame(self.parent_window, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Кнопки действий
        buttons_frame = tk.Frame(main_frame, bg='#f0f0f0')
        buttons_frame.pack(fill=tk.X, pady=10)
        
        # Кнопка "Добавить в базу"
        add_btn = tk.Button(
            buttons_frame,
            text="➕ Добавить в базу",
            font=("Arial", 12, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            command=self.add_to_database,
            cursor='hand2'
        )
        add_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка "Найти"
        search_btn = tk.Button(
            buttons_frame,
            text="🔍 Найти",
            font=("Arial", 12, "bold"),
            bg='#2196F3',
            fg='white',
            padx=20,
            pady=10,
            command=self.search_in_database,
            cursor='hand2'
        )
        search_btn.pack(side=tk.LEFT, padx=5)
        
        # Область для результатов
        results_frame = tk.LabelFrame(
            main_frame,
            text="Результаты",
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
    
    def add_to_database(self):
        """Открывает диалог для добавления лица в базу данных"""
        if self.face_features is None:
            messagebox.showwarning(
                "Предупреждение",
                "Сначала выполните анализ лица в окне анализа."
            )
            return
        
        # Создаем окно для ввода данных
        add_window = tk.Toplevel(self.parent_window)
        add_window.title("Добавить лицо в базу данных")
        add_window.geometry("500x400")
        add_window.configure(bg='#f0f0f0')
        add_window.transient(self.parent_window)
        add_window.grab_set()
        
        # Поля для ввода
        form_frame = tk.Frame(add_window, bg='#f0f0f0')
        form_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # ФИО
        tk.Label(
            form_frame,
            text="ФИО:",
            font=("Arial", 11, "bold"),
            bg='#f0f0f0',
            fg='#333'
        ).pack(anchor='w', pady=5)
        
        name_entry = tk.Entry(form_frame, font=("Arial", 11), width=40)
        name_entry.pack(fill=tk.X, pady=5)
        
        # Год рождения
        tk.Label(
            form_frame,
            text="Год рождения:",
            font=("Arial", 11, "bold"),
            bg='#f0f0f0',
            fg='#333'
        ).pack(anchor='w', pady=5)
        
        year_entry = tk.Entry(form_frame, font=("Arial", 11), width=40)
        year_entry.pack(fill=tk.X, pady=5)
        
        # Дополнительная информация
        tk.Label(
            form_frame,
            text="Дополнительная информация:",
            font=("Arial", 11, "bold"),
            bg='#f0f0f0',
            fg='#333'
        ).pack(anchor='w', pady=5)
        
        info_text = tk.Text(form_frame, font=("Arial", 10), width=40, height=8)
        info_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        def save_face():
            """Сохраняет лицо в базу данных"""
            full_name = name_entry.get().strip()
            
            if not full_name:
                messagebox.showerror("Ошибка", "Пожалуйста, введите ФИО.")
                return
            
            # Парсим год рождения
            birth_year = None
            year_str = year_entry.get().strip()
            if year_str:
                try:
                    birth_year = int(year_str)
                    if birth_year < 1900 or birth_year > 2100:
                        raise ValueError("Год должен быть в диапазоне 1900-2100")
                except ValueError as e:
                    messagebox.showerror("Ошибка", f"Неверный год рождения: {e}")
                    return
            
            # Получаем дополнительную информацию
            additional_info = info_text.get("1.0", tk.END).strip()
            if not additional_info:
                additional_info = None
            
            try:
                # Сохраняем в базу данных
                face_id = self.db.add_face(
                    full_name=full_name,
                    face_features=self.face_features,
                    birth_year=birth_year,
                    additional_info=additional_info
                )
                
                messagebox.showinfo(
                    "Успех",
                    f"Лицо успешно добавлено в базу данных!\nID: {face_id}"
                )
                
                add_window.destroy()
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось добавить лицо в базу данных:\n{e}")
        
        # Кнопка сохранения
        save_btn = tk.Button(
            form_frame,
            text="Сохранить",
            font=("Arial", 12, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            command=save_face,
            cursor='hand2'
        )
        save_btn.pack(pady=10)
        
        # Кнопка отмены
        cancel_btn = tk.Button(
            form_frame,
            text="Отмена",
            font=("Arial", 12),
            bg='#999',
            fg='white',
            padx=20,
            pady=10,
            command=add_window.destroy,
            cursor='hand2'
        )
        cancel_btn.pack(pady=5)
    
    def search_in_database(self):
        """Выполняет поиск лиц в базе данных"""
        if self.face_features is None:
            messagebox.showwarning(
                "Предупреждение",
                "Сначала выполните анализ лица в окне анализа."
            )
            return
        
        # Очищаем предыдущие результаты
        self.clear_results()
        
        # Показываем индикатор загрузки
        loading_label = tk.Label(
            self.results_frame,
            text="Выполняется поиск...",
            font=("Arial", 12),
            bg='white'
        )
        loading_label.pack(pady=20)
        self.parent_window.update()
        
        try:
            # Выполняем поиск с порогом 55%
            results = self.db.search_faces(self.face_features, threshold=55.0)
            
            loading_label.destroy()
            
            if not results:
                no_results_label = tk.Label(
                    self.results_frame,
                    text="Совпадений не найдено (порог: 55%)",
                    font=("Arial", 12),
                    bg='white',
                    fg='#666'
                )
                no_results_label.pack(pady=20)
                return
            
            # Заголовок результатов
            title_label = tk.Label(
                self.results_frame,
                text=f"Найдено совпадений: {len(results)}",
                font=("Arial", 14, "bold"),
                bg='white',
                fg='#333'
            )
            title_label.pack(pady=10)
            
            # Отображаем результаты
            for face_data, similarity in results:
                self.display_search_result(face_data, similarity)
            
        except Exception as e:
            loading_label.destroy()
            messagebox.showerror("Ошибка", f"Произошла ошибка при поиске:\n{e}")
        
        # Обновляем прокрутку
        self.results_canvas.update_idletasks()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        self.results_canvas.yview_moveto(0)
    
    def display_search_result(self, face_data: Dict, similarity: float):
        """
        Отображает результат поиска
        
        Args:
            face_data: Данные лица из базы данных
            similarity: Процент совпадения
        """
        result_frame = tk.LabelFrame(
            self.results_frame,
            text=f"Совпадение: {similarity:.1f}%",
            font=("Arial", 11, "bold"),
            bg='white',
            fg='#333',
            padx=10,
            pady=10
        )
        result_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # ФИО
        name_label = tk.Label(
            result_frame,
            text=f"ФИО: {face_data['full_name']}",
            font=("Arial", 11, "bold"),
            bg='white',
            fg='#333',
            anchor='w'
        )
        name_label.pack(fill=tk.X, pady=2)
        
        # Год рождения
        if face_data['birth_year']:
            year_label = tk.Label(
                result_frame,
                text=f"Год рождения: {face_data['birth_year']}",
                font=("Arial", 10),
                bg='white',
                fg='#666',
                anchor='w'
            )
            year_label.pack(fill=tk.X, pady=2)
        
        # Дополнительная информация
        if face_data['additional_info']:
            info_label = tk.Label(
                result_frame,
                text=f"Доп. информация: {face_data['additional_info']}",
                font=("Arial", 10),
                bg='white',
                fg='#666',
                anchor='w',
                wraplength=800,
                justify='left'
            )
            info_label.pack(fill=tk.X, pady=2)
        
        # Дата добавления
        if face_data.get('created_at'):
            date_label = tk.Label(
                result_frame,
                text=f"Добавлено: {face_data['created_at']}",
                font=("Arial", 9),
                bg='white',
                fg='#999',
                anchor='w'
            )
            date_label.pack(fill=tk.X, pady=2)
        
        # Цвет рамки в зависимости от процента совпадения
        if similarity >= 80:
            result_frame.configure(fg='#4CAF50')  # Зеленый
        elif similarity >= 65:
            result_frame.configure(fg='#FF9800')  # Оранжевый
        else:
            result_frame.configure(fg='#2196F3')  # Синий
    
    def clear_results(self):
        """Очищает область результатов"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()

