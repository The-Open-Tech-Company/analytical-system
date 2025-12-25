"""
Analytical System - Main Application
Main application skeleton for analytical system

Repository: https://github.com/The-Open-Tech-Company/analytical-system
License: Unlicense (Open Source)
"""
import tkinter as tk
from tkinter import messagebox
from face_comparison_window import FaceComparisonWindow
from face_analysis_window import FaceAnalysisWindow
from voice_comparison_window import VoiceComparisonWindow
from voice_analysis_window import VoiceAnalysisWindow
from settings_window import SettingsWindow


class MainApplication:
    """Главное окно приложения - скелет программы"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Аналитическая система")
        self.root.geometry("800x600")
        self.root.configure(bg='white')
        
        # Список открытых окон для правильного управления
        self.open_windows = []
        
        # Создаем интерфейс
        self.create_widgets()
        
        # Обработчик закрытия окна
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """Создает виджеты главного окна"""
        
        # Верхняя панель с кнопками
        top_frame = tk.Frame(self.root, bg='white', height=50)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        top_frame.pack_propagate(False)
        
        # Кнопка "Лицо" с выпадающим меню
        face_btn = tk.Menubutton(
            top_frame,
            text="Лицо",
            font=("Arial", 12, "bold"),
            bg='#2196F3',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=2
        )
        face_btn.pack(side=tk.LEFT, padx=5)
        
        face_menu = tk.Menu(face_btn, tearoff=0)
        face_menu.add_command(
            label="Анализ лица",
            command=self.open_face_analysis,
            font=("Arial", 10)
        )
        face_menu.add_command(
            label="Сравнить лица",
            command=self.open_face_comparison,
            font=("Arial", 10)
        )
        face_btn.config(menu=face_menu)
        
        # Кнопка "Голос" с выпадающим меню
        voice_btn = tk.Menubutton(
            top_frame,
            text="Голос",
            font=("Arial", 12, "bold"),
            bg='#9C27B0',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=2
        )
        voice_btn.pack(side=tk.LEFT, padx=5)
        
        voice_menu = tk.Menu(voice_btn, tearoff=0)
        voice_menu.add_command(
            label="Анализ голоса",
            command=self.open_voice_analysis,
            font=("Arial", 10)
        )
        voice_menu.add_command(
            label="Сравнение голоса",
            command=self.open_voice_comparison,
            font=("Arial", 10)
        )
        voice_btn.config(menu=voice_menu)
        
        # Кнопка "Настройки"
        settings_btn = tk.Button(
            top_frame,
            text="Настройки",
            font=("Arial", 12, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=2,
            command=self.open_settings
        )
        settings_btn.pack(side=tk.LEFT, padx=5)
        
        # Основная область с приветствием
        main_frame = tk.Frame(self.root, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        welcome_label = tk.Label(
            main_frame,
            text="Аналитическая система.\nДобро пожаловать",
            font=("Arial", 24, "bold"),
            bg='white',
            fg='#333',
            justify=tk.CENTER
        )
        welcome_label.pack(expand=True)
    
    def open_face_analysis(self):
        """Открывает окно анализа лица"""
        try:
            window = tk.Toplevel(self.root)
            window.transient(self.root)  # Делаем окно дочерним (не будет перекрывать главное)
            
            app = FaceAnalysisWindow(window, self)
            self.open_windows.append(window)
            
            # Обработчик закрытия окна
            def on_close():
                if window in self.open_windows:
                    self.open_windows.remove(window)
                window.destroy()
            
            window.protocol("WM_DELETE_WINDOW", on_close)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть окно анализа лица: {e}")
    
    def open_face_comparison(self):
        """Открывает окно сравнения лиц"""
        try:
            window = tk.Toplevel(self.root)
            window.transient(self.root)  # Делаем окно дочерним (не будет перекрывать главное)
            
            app = FaceComparisonWindow(window, self)
            self.open_windows.append(window)
            
            # Обработчик закрытия окна
            def on_close():
                if window in self.open_windows:
                    self.open_windows.remove(window)
                window.destroy()
            
            window.protocol("WM_DELETE_WINDOW", on_close)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть окно сравнения лиц: {e}")
    
    def open_voice_analysis(self):
        """Открывает окно анализа голоса"""
        try:
            window = tk.Toplevel(self.root)
            window.transient(self.root)  # Делаем окно дочерним (не будет перекрывать главное)
            
            app = VoiceAnalysisWindow(window, self)
            self.open_windows.append(window)
            
            # Обработчик закрытия окна
            def on_close():
                if window in self.open_windows:
                    self.open_windows.remove(window)
                window.destroy()
            
            window.protocol("WM_DELETE_WINDOW", on_close)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть окно анализа голоса: {e}")
    
    def open_voice_comparison(self):
        """Открывает окно сравнения голосов"""
        try:
            window = tk.Toplevel(self.root)
            window.transient(self.root)  # Делаем окно дочерним (не будет перекрывать главное)
            
            app = VoiceComparisonWindow(window, self)
            self.open_windows.append(window)
            
            # Обработчик закрытия окна
            def on_close():
                if window in self.open_windows:
                    self.open_windows.remove(window)
                window.destroy()
            
            window.protocol("WM_DELETE_WINDOW", on_close)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть окно сравнения голосов: {e}")
    
    def open_settings(self):
        """Открывает окно настроек"""
        try:
            window = tk.Toplevel(self.root)
            window.transient(self.root)  # Делаем окно дочерним (не будет перекрывать главное)
            
            app = SettingsWindow(window, self)
            self.open_windows.append(window)
            
            # Обработчик закрытия окна
            def on_close():
                if window in self.open_windows:
                    self.open_windows.remove(window)
                window.destroy()
            
            window.protocol("WM_DELETE_WINDOW", on_close)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть окно настроек: {e}")
    
    def on_closing(self):
        """Обработчик закрытия главного окна"""
        for window in self.open_windows[:]:
            try:
                window.destroy()
            except (tk.TclError, AttributeError):
                pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main():
    """Запуск главного приложения"""
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()
