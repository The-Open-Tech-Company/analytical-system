"""
Модуль для визуализации голосовых характеристик
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Tuple, Optional
import librosa
import io
from PIL import Image


class VoiceVisualizer:
    """Класс для визуализации голосовых данных"""
    
    def __init__(self):
        """Инициализация визуализатора"""
        self.fig_size = (10, 6)  # Уменьшено для лучшего отображения
        self.dpi = 80  # Уменьшено для меньшего размера изображений
    
    def create_spectrogram(self, audio: np.ndarray, sr: float) -> np.ndarray:
        """
        Создает спектрограмму аудио
        
        Args:
            audio: Аудио данные
            sr: Частота дискретизации
            
        Returns:
            Изображение спектрограммы как numpy array
        """
        # Вычисляем спектрограмму
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Преобразуем в децибелы
        db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Отображаем спектрограмму
        try:
            img = librosa.display.specshow(
                db,
                x_axis='time',
                y_axis='hz',
                sr=sr,
                ax=ax,
                cmap='viridis'
            )
            
            ax.set_title('Спектрограмма', fontsize=14, fontweight='bold')
            ax.set_xlabel('Время (сек)', fontsize=12)
            ax.set_ylabel('Частота (Hz)', fontsize=12)
            
            # Добавляем цветовую шкалу
            try:
                if img is not None:
                    plt.colorbar(img, ax=ax, format='%+2.0f dB')
            except Exception:
                # Альтернативный способ создания colorbar
                im = ax.imshow(db, aspect='auto', origin='lower', cmap='viridis', 
                              extent=[0, len(audio)/sr, 0, sr/2])
                plt.colorbar(im, ax=ax, format='%+2.0f dB')
        except Exception:
            # Если specshow не работает, используем imshow
            times = np.linspace(0, len(audio)/sr, db.shape[1])
            freqs = np.linspace(0, sr/2, db.shape[0])
            im = ax.imshow(db, aspect='auto', origin='lower', cmap='viridis', 
                          extent=[times[0], times[-1], freqs[0], freqs[-1]])
            ax.set_title('Спектрограмма', fontsize=14, fontweight='bold')
            ax.set_xlabel('Время (сек)', fontsize=12)
            ax.set_ylabel('Частота (Hz)', fontsize=12)
            plt.colorbar(im, ax=ax, format='%+2.0f dB')
        
        # Конвертируем в numpy array
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
        buf.seek(0)
        img_array = np.array(Image.open(buf))
        plt.close(fig)
        
        return img_array
    
    def create_volume_plot(self, audio: np.ndarray, sr: float) -> np.ndarray:
        """
        Создает график громкости (RMS) во времени
        
        Args:
            audio: Аудио данные
            sr: Частота дискретизации
            
        Returns:
            Изображение графика как numpy array
        """
        # Вычисляем RMS (громкость)
        rms = librosa.feature.rms(y=audio)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Рисуем график
        ax.plot(times, rms, color='#2196F3', linewidth=2)
        ax.fill_between(times, rms, alpha=0.3, color='#2196F3')
        
        ax.set_title('Громкость (RMS)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Время (сек)', fontsize=12)
        ax.set_ylabel('Громкость', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Конвертируем в numpy array
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
        buf.seek(0)
        img_array = np.array(Image.open(buf))
        plt.close(fig)
        
        return img_array
    
    def create_pitch_plot(self, audio: np.ndarray, sr: float) -> np.ndarray:
        """
        Создает график интонаций (основной частоты) во времени
        
        Args:
            audio: Аудио данные
            sr: Частота дискретизации
            
        Returns:
            Изображение графика как numpy array
        """
        # Вычисляем основную частоту (pitch)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        
        # Извлекаем значения pitch
        pitch_values = []
        times = []
        frame_times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr)
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
                times.append(frame_times[t])
        
        if len(pitch_values) == 0:
            # Если нет данных, создаем пустой график
            pitch_values = [0]
            times = [0]
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Рисуем график
        ax.plot(times, pitch_values, color='#4CAF50', linewidth=2, marker='o', markersize=3)
        
        ax.set_title('Интонации (Основная частота)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Время (сек)', fontsize=12)
        ax.set_ylabel('Частота (Hz)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Конвертируем в numpy array
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
        buf.seek(0)
        img_array = np.array(Image.open(buf))
        plt.close(fig)
        
        return img_array
    
    def create_frequency_plot(self, audio: np.ndarray, sr: float) -> np.ndarray:
        """
        Создает график частотного спектра
        
        Args:
            audio: Аудио данные
            sr: Частота дискретизации
            
        Returns:
            Изображение графика как numpy array
        """
        # Вычисляем FFT
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        frequencies = np.fft.fftfreq(len(fft), 1/sr)
        
        # Берем только положительные частоты
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        magnitude = magnitude[positive_freq_idx]
        
        # Ограничиваем до 5000 Hz для лучшей визуализации
        max_freq_idx = frequencies <= 5000
        frequencies = frequencies[max_freq_idx]
        magnitude = magnitude[max_freq_idx]
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Рисуем график
        ax.plot(frequencies, magnitude, color='#FF9800', linewidth=2)
        ax.fill_between(frequencies, magnitude, alpha=0.3, color='#FF9800')
        
        ax.set_title('Частотный спектр', fontsize=14, fontweight='bold')
        ax.set_xlabel('Частота (Hz)', fontsize=12)
        ax.set_ylabel('Амплитуда', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Конвертируем в numpy array
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
        buf.seek(0)
        img_array = np.array(Image.open(buf))
        plt.close(fig)
        
        return img_array
    
    def create_comprehensive_visualization(self, audio: np.ndarray, sr: float, 
                                          features: Dict) -> np.ndarray:
        """
        Создает комплексную визуализацию со всеми графиками
        
        Args:
            audio: Аудио данные
            sr: Частота дискретизации
            features: Словарь с характеристиками
            
        Returns:
            Изображение как numpy array
        """
        # Создаем фигуру с несколькими подграфиками (уменьшенный размер)
        fig = plt.figure(figsize=(12, 8), dpi=self.dpi)
        
        # Спектрограмма
        ax1 = plt.subplot(2, 2, 1)
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        db = librosa.amplitude_to_db(magnitude, ref=np.max)
        try:
            img1 = librosa.display.specshow(db, x_axis='time', y_axis='hz', sr=sr, ax=ax1, cmap='viridis')
            ax1.set_title('Спектрограмма', fontsize=12, fontweight='bold')
            try:
                if img1 is not None:
                    plt.colorbar(img1, ax=ax1, format='%+2.0f dB')
            except Exception:
                # Альтернативный способ
                im = ax1.imshow(db, aspect='auto', origin='lower', cmap='viridis', 
                              extent=[0, len(audio)/sr, 0, sr/2])
                plt.colorbar(im, ax=ax1, format='%+2.0f dB')
        except Exception:
            # Если specshow не работает, используем imshow
            times = np.linspace(0, len(audio)/sr, db.shape[1])
            freqs = np.linspace(0, sr/2, db.shape[0])
            im = ax1.imshow(db, aspect='auto', origin='lower', cmap='viridis', 
                          extent=[times[0], times[-1], freqs[0], freqs[-1]])
            ax1.set_title('Спектрограмма', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Время (сек)')
            ax1.set_ylabel('Частота (Hz)')
            plt.colorbar(im, ax=ax1, format='%+2.0f dB')
        
        # Громкость
        ax2 = plt.subplot(2, 2, 2)
        rms = librosa.feature.rms(y=audio)[0]
        times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        ax2.plot(times_rms, rms, color='#2196F3', linewidth=2)
        ax2.fill_between(times_rms, rms, alpha=0.3, color='#2196F3')
        ax2.set_title('Громкость (RMS)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Время (сек)')
        ax2.set_ylabel('Громкость')
        ax2.grid(True, alpha=0.3)
        
        # Интонации
        ax3 = plt.subplot(2, 2, 3)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        times_pitch = []
        frame_times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr)
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
                times_pitch.append(frame_times[t])
        if len(pitch_values) > 0:
            ax3.plot(times_pitch, pitch_values, color='#4CAF50', linewidth=2, marker='o', markersize=2)
        ax3.set_title('Интонации (Основная частота)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Время (сек)')
        ax3.set_ylabel('Частота (Hz)')
        ax3.grid(True, alpha=0.3)
        
        # Частотный спектр
        ax4 = plt.subplot(2, 2, 4)
        fft = np.fft.fft(audio)
        magnitude_fft = np.abs(fft)
        frequencies = np.fft.fftfreq(len(fft), 1/sr)
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        magnitude_fft = magnitude_fft[positive_freq_idx]
        max_freq_idx = frequencies <= 5000
        frequencies = frequencies[max_freq_idx]
        magnitude_fft = magnitude_fft[max_freq_idx]
        ax4.plot(frequencies, magnitude_fft, color='#FF9800', linewidth=2)
        ax4.fill_between(frequencies, magnitude_fft, alpha=0.3, color='#FF9800')
        ax4.set_title('Частотный спектр', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Частота (Hz)')
        ax4.set_ylabel('Амплитуда')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Конвертируем в numpy array
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=self.dpi)
        buf.seek(0)
        img_array = np.array(Image.open(buf))
        plt.close(fig)
        
        return img_array

