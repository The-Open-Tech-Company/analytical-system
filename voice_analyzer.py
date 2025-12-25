"""
Analytical System - Voice Analyzer Module
Voice analysis: gender, accent, emotion detection

Repository: https://github.com/The-Open-Tech-Company/analytical-system
License: Unlicense (Open Source)
"""
import numpy as np
import librosa
import os
import tempfile
from typing import Dict, Optional, Tuple
import soundfile as sf
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False


class VoiceAnalyzer:
    """Класс для анализа голосовых характеристик"""
    
    def __init__(self):
        """Инициализация анализатора голоса"""
        # Инициализация распознавателя речи
        self.recognizer = None
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                # Настройка параметров для лучшего распознавания
                self.recognizer.energy_threshold = 300  # Порог энергии для обнаружения речи
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.8  # Пауза между фразами
            except Exception as e:
                print(f"Предупреждение: не удалось инициализировать SpeechRecognition: {e}")
                self.recognizer = None
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, float]:
        """
        Загружает аудиофайл
        
        Args:
            file_path: Путь к аудиофайлу
            
        Returns:
            Кортеж (аудио данные, частота дискретизации)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        try:
            # Пробуем загрузить через librosa (автоматически конвертирует в моно и 22050 Hz)
            audio, sr = librosa.load(file_path, sr=None)
            return audio, sr
        except Exception as e:
            # Если librosa не работает, пробуем soundfile
            try:
                audio, sr = sf.read(file_path)
                # Конвертируем в моно, если стерео
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                return audio, sr
            except Exception as e2:
                raise ValueError(f"Не удалось загрузить аудиофайл: {e}, {e2}")
    
    def extract_features(self, audio: np.ndarray, sr: float) -> Dict:
        """
        Извлекает характеристики из аудио
        
        Args:
            audio: Аудио данные
            sr: Частота дискретизации
            
        Returns:
            Словарь с характеристиками
        """
        # Фильтрация помех: удаляем короткие всплески (хлопки) и высокочастотные шумы (скрипы)
        audio_filtered = audio.copy()
        
        # 1. Удаление коротких всплесков (хлопки) - используем медианный фильтр
        from scipy import signal
        try:
            # Медианный фильтр для удаления коротких всплесков
            window_size = int(sr * 0.01)  # 10 мс окно
            if window_size % 2 == 0:
                window_size += 1
            if window_size > 1:
                audio_filtered = signal.medfilt(audio_filtered, kernel_size=window_size)
        except:
            pass  # Если scipy недоступен, пропускаем фильтрацию
        
        # 2. Удаление высокочастотных шумов (скрипы) - низкочастотный фильтр
        try:
            # Простой высокочастотный фильтр для удаления шумов выше 8000 Hz
            # Используем простой фильтр через FFT
            fft = np.fft.fft(audio_filtered)
            freqs = np.fft.fftfreq(len(audio_filtered), 1/sr)
            # Подавляем частоты выше 8000 Hz (скрипы обычно выше)
            fft[np.abs(freqs) > 8000] *= 0.1
            audio_filtered = np.real(np.fft.ifft(fft))
        except:
            pass
        
        # 3. Нормализация для уменьшения влияния помех
        if np.max(np.abs(audio_filtered)) > 0:
            audio_filtered = audio_filtered / np.max(np.abs(audio_filtered)) * 0.9
        
        # Используем отфильтрованное аудио для извлечения характеристик
        audio = audio_filtered
        
        features = {}
        
        # Основная частота (F0) - pitch (улучшенный метод)
        # Используем несколько методов для более точного определения
        
        # Метод 1: pyin (более точный, но медленнее)
        try:
            f0_pyin, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7')   # ~2093 Hz
            )
            # Берем только voiced (озвученные) сегменты
            pitch_values_pyin = f0_pyin[voiced_flag > 0.5]
            pitch_values_pyin = pitch_values_pyin[pitch_values_pyin > 0]
        except:
            pitch_values_pyin = np.array([])
        
        # Метод 2: piptrack (быстрый, но менее точный)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values_piptrack = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values_piptrack.append(pitch)
        pitch_values_piptrack = np.array(pitch_values_piptrack)
        
        # Объединяем результаты обоих методов, предпочитая pyin
        if len(pitch_values_pyin) > 0:
            pitch_values = pitch_values_pyin
        elif len(pitch_values_piptrack) > 0:
            pitch_values = pitch_values_piptrack
        else:
            pitch_values = np.array([])
        
        # Фильтруем выбросы (удаляем значения вне разумного диапазона 50-500 Hz)
        if len(pitch_values) > 0:
            pitch_values = pitch_values[(pitch_values >= 50) & (pitch_values <= 500)]
        
        if len(pitch_values) > 0:
            # Используем медиану для более устойчивой оценки (менее чувствительна к выбросам)
            features['pitch_median'] = float(np.median(pitch_values))
            features['pitch_mean'] = float(np.mean(pitch_values))  # Среднее для совместимости
            features['pitch_std'] = float(np.std(pitch_values))
            features['pitch_min'] = float(np.percentile(pitch_values, 10))  # 10-й перцентиль
            features['pitch_max'] = float(np.percentile(pitch_values, 90))  # 90-й перцентиль
        else:
            features['pitch_median'] = 0.0
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_min'] = 0.0
            features['pitch_max'] = 0.0
        
        # MFCC (Mel-frequency cepstral coefficients) - характеристики тембра
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = [float(x) for x in np.mean(mfccs, axis=1)]
        features['mfcc_std'] = [float(x) for x in np.std(mfccs, axis=1)]
        
        # Спектральные характеристики
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        # Zero crossing rate - частота пересечения нуля
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # Энергия (RMS)
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        features['rms_max'] = float(np.max(rms))
        
        # Форманты (приблизительно через спектральные пики)
        # F1 и F2 - основные форманты для определения гласных
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Находим пики в спектре для определения формант
        # Упрощенный метод - берем частоты с максимальной энергией в разных диапазонах
        freqs = librosa.fft_frequencies(sr=sr)
        
        # F1 обычно в диапазоне 300-1000 Hz
        f1_range = (freqs >= 300) & (freqs <= 1000)
        if np.any(f1_range):
            f1_freqs = freqs[f1_range]
            f1_energy = np.mean(magnitude[f1_range, :], axis=1)
            f1_max_idx = np.argmax(f1_energy)
            if f1_max_idx < len(f1_freqs):
                features['formant_f1'] = float(f1_freqs[f1_max_idx])
            else:
                features['formant_f1'] = 0.0
        else:
            features['formant_f1'] = 0.0
        
        # F2 обычно в диапазоне 1000-3000 Hz
        f2_range = (freqs >= 1000) & (freqs <= 3000)
        if np.any(f2_range):
            f2_freqs = freqs[f2_range]
            f2_energy = np.mean(magnitude[f2_range, :], axis=1)
            f2_max_idx = np.argmax(f2_energy)
            if f2_max_idx < len(f2_freqs):
                features['formant_f2'] = float(f2_freqs[f2_max_idx])
            else:
                features['formant_f2'] = 0.0
        else:
            features['formant_f2'] = 0.0
        
        # Длительность
        features['duration'] = float(len(audio) / sr)
        
        # Темп речи (приблизительно через количество слогов)
        # Упрощенный метод - используем количество пиков энергии
        energy = librosa.feature.rms(y=audio)[0]
        energy_threshold = np.mean(energy) * 0.5
        peaks = np.where(energy > energy_threshold)[0]
        if len(peaks) > 0:
            # Группируем близкие пики
            peak_groups = []
            current_group = [peaks[0]]
            for i in range(1, len(peaks)):
                if peaks[i] - peaks[i-1] < 10:  # Пики близко друг к другу
                    current_group.append(peaks[i])
                else:
                    peak_groups.append(current_group)
                    current_group = [peaks[i]]
            peak_groups.append(current_group)
            features['speech_rate'] = float(len(peak_groups) / features['duration']) if features['duration'] > 0 else 0.0
        else:
            features['speech_rate'] = 0.0
        
        return features
    
    def detect_gender(self, features: Dict) -> Tuple[str, float]:
        """
        Определяет пол по характеристикам голоса (адаптировано для RU региона)
        
        Args:
            features: Словарь с характеристиками
            
        Returns:
            Кортеж (пол, уверенность)
        """
        # Для русского языка характерны следующие диапазоны:
        # Мужские голоса: 80-180 Hz (медиана ~120 Hz)
        # Женские голоса: 150-300 Hz (медиана ~220 Hz)
        # Детские голоса: 250-400 Hz
        
        pitch_mean = features.get('pitch_mean', 0)
        pitch_median = features.get('pitch_median', 0)
        pitch_std = features.get('pitch_std', 0)
        formant_f1 = features.get('formant_f1', 0)
        formant_f2 = features.get('formant_f2', 0)
        spectral_centroid = features.get('spectral_centroid_mean', 0)
        
        # Используем медиану, если доступна (более устойчива к выбросам)
        pitch = pitch_median if pitch_median > 0 else pitch_mean
        
        # Если нет данных о pitch, используем форманты и спектральный центроид
        if pitch == 0:
            if formant_f1 > 0:
                # Мужские голоса имеют более низкие форманты F1 (обычно 400-700 Hz)
                # Женские голоса имеют более высокие форманты F1 (обычно 600-900 Hz)
                if formant_f1 < 550:
                    return ("Мужской", 0.75)
                elif formant_f1 > 750:
                    return ("Женский", 0.75)
                else:
                    # Пограничная зона
                    if spectral_centroid > 0 and spectral_centroid < 2000:
                        return ("Мужской", 0.60)
                    elif spectral_centroid > 2500:
                        return ("Женский", 0.60)
                    else:
                        return ("Не определен", 0.3)
            else:
                return ("Не определен", 0.0)
        
        # Определяем по основной частоте с учетом RU региона
        # Четкая граница между мужскими и женскими голосами: ~165 Hz
        
        if pitch < 100:
            # Очень низкий pitch - определенно мужской
            confidence = min(0.98, 0.7 + (100 - pitch) / 50)
            return ("Мужской", confidence)
        elif pitch < 140:
            # Низкий pitch - вероятно мужской
            confidence = 0.75 + (140 - pitch) / 80 * 0.2
            return ("Мужской", confidence)
        elif pitch < 165:
            # Пограничная зона (140-165 Hz) - используем дополнительные характеристики
            if formant_f1 > 0:
                if formant_f1 < 600:
                    return ("Мужской", 0.65)
                else:
                    return ("Женский", 0.60)
            elif spectral_centroid > 0:
                if spectral_centroid < 2200:
                    return ("Мужской", 0.60)
                else:
                    return ("Женский", 0.60)
            else:
                # По умолчанию в пограничной зоне склоняемся к мужскому
                return ("Мужской", 0.55)
        elif pitch < 200:
            # Пограничная зона (165-200 Hz) - используем дополнительные характеристики
            if formant_f1 > 0:
                if formant_f1 > 700:
                    return ("Женский", 0.70)
                elif formant_f1 < 550:
                    return ("Мужской", 0.65)
                else:
                    # Средние значения формант
                    if pitch < 180:
                        return ("Мужской", 0.55)
                    else:
                        return ("Женский", 0.60)
            elif spectral_centroid > 0:
                if spectral_centroid > 2500:
                    return ("Женский", 0.65)
                elif spectral_centroid < 2000:
                    return ("Мужской", 0.60)
                else:
                    if pitch < 180:
                        return ("Мужской", 0.55)
                    else:
                        return ("Женский", 0.60)
            else:
                # По умолчанию в пограничной зоне
                if pitch < 180:
                    return ("Мужской", 0.52)
                else:
                    return ("Женский", 0.58)
        elif pitch < 250:
            # Средний pitch - вероятно женский
            confidence = 0.75 + (pitch - 200) / 50 * 0.2
            return ("Женский", confidence)
        else:
            # Высокий pitch - определенно женский
            confidence = min(0.98, 0.7 + (pitch - 250) / 100)
            return ("Женский", confidence)
    
    def detect_accent(self, features: Dict, language: str = None) -> Tuple[str, float]:
        """
        Определяет акцент (улучшенная версия)
        
        Args:
            features: Словарь с характеристиками
            language: Определенный язык (опционально, для более точного определения)
            
        Returns:
            Кортеж (акцент, уверенность)
        """
        # Используем характеристики формант и MFCC для определения
        
        mfcc_mean = features.get('mfcc_mean', [])
        formant_f1 = features.get('formant_f1', 0)
        formant_f2 = features.get('formant_f2', 0)
        spectral_centroid = features.get('spectral_centroid_mean', 0)
        
        if len(mfcc_mean) == 0:
            return ("Не определен", 0.0)
        
        # Если язык определен как русский, проверяем, является ли это носителем
        if language == 'Русский':
            # Для русского языка проверяем, соответствует ли речь характеристикам носителя
            russian_native_score = 0.0
            
            if formant_f1 > 0 and formant_f2 > 0:
                f2_f1_ratio = formant_f2 / formant_f1
                # Характеристики носителя русского языка
                if 1.8 < f2_f1_ratio < 2.3:
                    russian_native_score += 0.4
                if 450 < formant_f1 < 650:
                    russian_native_score += 0.3
                if 1200 < formant_f2 < 1800:
                    russian_native_score += 0.3
            
            if len(mfcc_mean) >= 2:
                mfcc_0 = mfcc_mean[0]
                if -13 < mfcc_0 < -7:
                    russian_native_score += 0.2
            
            if 1700 < spectral_centroid < 2300:
                russian_native_score += 0.2
            
            # Если характеристики очень близки к носителю, возвращаем "Нет акцента"
            if russian_native_score > 0.7:
                return ("Нет акцента", min(0.85, russian_native_score))
            elif russian_native_score > 0.4:
                # Характеристики близки, но не идеальны
                return ("Русский акцент", 0.6)
            else:
                # Характеристики отличаются от носителя
                return ("Русский акцент", 0.5)
        
        # Для других языков используем общую логику
        if formant_f1 > 0 and formant_f2 > 0:
            f2_f1_ratio = formant_f2 / formant_f1
            
            # Примерные диапазоны для разных акцентов
            if 1.5 < f2_f1_ratio < 2.5:
                return ("Русский акцент", 0.5)
            elif f2_f1_ratio > 2.5:
                return ("Английский акцент", 0.5)
            else:
                return ("Другой акцент", 0.4)
        else:
            # Используем MFCC для приблизительной оценки
            mfcc_1 = mfcc_mean[0] if len(mfcc_mean) > 0 else 0
            if mfcc_1 < -5:
                return ("Русский акцент", 0.4)
            else:
                return ("Другой акцент", 0.3)
    
    def detect_emotion(self, features: Dict) -> Tuple[str, float]:
        """
        Определяет эмоции по характеристикам голоса
        
        Args:
            features: Словарь с характеристиками
            
        Returns:
            Кортеж (эмоция, уверенность)
        """
        # Упрощенная версия определения эмоций
        # В реальности нужна обученная модель
        
        pitch_mean = features.get('pitch_mean', 0)
        pitch_std = features.get('pitch_std', 0)
        rms_mean = features.get('rms_mean', 0)
        rms_std = features.get('rms_std', 0)
        zcr_mean = features.get('zcr_mean', 0)
        speech_rate = features.get('speech_rate', 0)
        
        if pitch_mean == 0:
            return ("Нейтральная", 0.3)
        
        # Высокая вариативность pitch и высокая громкость - радость/возбуждение
        if pitch_std > 50 and rms_mean > 0.1:
            return ("Радость/Возбуждение", 0.6)
        
        # Низкий pitch и низкая вариативность - грусть
        if pitch_mean < 120 and pitch_std < 20:
            return ("Грусть", 0.6)
        
        # Высокий pitch и высокая вариативность - страх/тревога
        if pitch_mean > 200 and pitch_std > 40:
            return ("Страх/Тревога", 0.6)
        
        # Высокая громкость и высокая частота пересечения нуля - гнев
        if rms_mean > 0.15 and zcr_mean > 0.1:
            return ("Гнев", 0.6)
        
        # Быстрый темп речи - возбуждение
        if speech_rate > 5:
            return ("Возбуждение", 0.5)
        
        # По умолчанию
        return ("Нейтральная", 0.5)
    
    def _detect_language_with_speech_recognition(self, audio_file: str) -> Tuple[Optional[str], float]:
        """
        Определяет язык с помощью SpeechRecognition (улучшенная версия)
        Использует сравнение качества распознавания для разных языков
        
        Args:
            audio_file: Путь к аудиофайлу
            
        Returns:
            Кортеж (язык, уверенность) или (None, 0.0) при ошибке
        """
        if not SPEECH_RECOGNITION_AVAILABLE or self.recognizer is None:
            return None, 0.0
        
        # Маппинг кодов языков на русские названия
        language_map = {
            'ru-RU': 'Русский',
            'ru': 'Русский',
            'en-US': 'Английский',
            'en-GB': 'Английский',
            'en': 'Английский',
            'es-ES': 'Испанский',
            'es': 'Испанский',
            'de-DE': 'Немецкий',
            'de': 'Немецкий',
            'fr-FR': 'Французский',
            'fr': 'Французский',
            'it-IT': 'Итальянский',
            'it': 'Итальянский',
            'pt-PT': 'Португальский',
            'pt': 'Португальский',
            'pl-PL': 'Польский',
            'pl': 'Польский',
            'uk-UA': 'Украинский',
            'uk': 'Украинский',
        }
        
        # Расширенный список языков для проверки (русский - приоритетный)
        languages_to_check = ['ru-RU', 'en-US', 'es-ES', 'de-DE', 'fr-FR', 'it-IT', 'pl-PL', 'uk-UA']
        
        try:
            # Загружаем аудиофайл
            with sr.AudioFile(audio_file) as source:
                # Подстраиваем уровень шума (увеличиваем время для лучшей адаптации)
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                # Читаем все аудио
                audio_data = self.recognizer.record(source)
            
            # Словарь для хранения результатов по каждому языку
            language_results = {}
            
            # Пробуем распознать с каждым языком и оцениваем качество
            for lang_code in languages_to_check:
                lang_name = language_map.get(lang_code, lang_code)
                language_results[lang_name] = {
                    'confidence': 0.0,
                    'text_length': 0,
                    'word_count': 0,
                    'success': False,
                    'errors': 0
                }
                
                # Пробуем несколько раз для надежности
                for attempt in range(2):
                    try:
                        # Пробуем распознать с указанием языка и получить детальную информацию
                        result = self.recognizer.recognize_google(
                            audio_data, 
                            language=lang_code, 
                            show_all=True
                        )
                        
                        if result:
                            # Обрабатываем результат
                            if isinstance(result, dict) and 'alternative' in result:
                                alternatives = result.get('alternative', [])
                                if alternatives:
                                    # Берем лучшую альтернативу
                                    best_alt = alternatives[0]
                                    text = best_alt.get('transcript', '')
                                    confidence = best_alt.get('confidence', 0.5)
                                    
                                    if text and len(text.strip()) > 0:
                                        # Подсчитываем слова
                                        words = text.strip().split()
                                        word_count = len(words)
                                        text_length = len(text)
                                        
                                        # Обновляем результаты (берем лучший результат из попыток)
                                        if confidence > language_results[lang_name]['confidence']:
                                            language_results[lang_name]['confidence'] = confidence
                                            language_results[lang_name]['text_length'] = text_length
                                            language_results[lang_name]['word_count'] = word_count
                                            language_results[lang_name]['success'] = True
                                        
                                        # Если уверенность высокая, можем остановиться
                                        if confidence > 0.8:
                                            break
                            
                            elif isinstance(result, str) and len(result.strip()) > 0:
                                # Если результат - строка
                                text = result.strip()
                                words = text.split()
                                word_count = len(words)
                                text_length = len(text)
                                
                                # Даем базовую уверенность 0.7 для успешного распознавания
                                if language_results[lang_name]['confidence'] < 0.7:
                                    language_results[lang_name]['confidence'] = 0.7
                                    language_results[lang_name]['text_length'] = text_length
                                    language_results[lang_name]['word_count'] = word_count
                                    language_results[lang_name]['success'] = True
                        
                    except sr.UnknownValueError:
                        # Не удалось распознать - увеличиваем счетчик ошибок
                        language_results[lang_name]['errors'] += 1
                    except sr.RequestError:
                        # Ошибка API - пропускаем этот язык
                        break
                    except Exception:
                        # Другие ошибки
                        language_results[lang_name]['errors'] += 1
            
            # Вычисляем финальные оценки для каждого языка
            final_scores = {}
            
            for lang_name, results in language_results.items():
                if not results['success']:
                    # Если не удалось распознать, даем очень низкую оценку
                    final_scores[lang_name] = 0.0
                    continue
                
                # Базовый score на основе уверенности
                score = results['confidence']
                
                # Бонус за длину текста (больше текста = лучше распознавание)
                if results['text_length'] > 10:
                    text_bonus = min(0.2, results['text_length'] / 100.0)
                    score += text_bonus
                
                # Бонус за количество слов (больше слов = лучше)
                if results['word_count'] > 2:
                    word_bonus = min(0.15, results['word_count'] / 20.0)
                    score += word_bonus
                
                # Штраф за ошибки распознавания
                if results['errors'] > 0:
                    error_penalty = min(0.3, results['errors'] * 0.1)
                    score = max(0.0, score - error_penalty)
                
                # Ограничиваем score диапазоном [0, 1]
                final_scores[lang_name] = min(1.0, max(0.0, score))
            
            # Если нет успешных результатов, возвращаем None
            if not any(score > 0.0 for score in final_scores.values()):
                return None, 0.0
            
            # Находим язык с максимальным score
            best_lang = max(final_scores, key=final_scores.get)
            best_score = final_scores[best_lang]
            
            # Проверяем, что лучший результат значительно лучше остальных
            sorted_scores = sorted(final_scores.values(), reverse=True)
            if len(sorted_scores) > 1:
                # Если разница между первым и вторым меньше 0.15, снижаем уверенность
                if sorted_scores[0] - sorted_scores[1] < 0.15:
                    best_score *= 0.85
            
            # Нормализуем уверенность (0.0-1.0 -> 0.6-0.95)
            # Минимальная уверенность 0.6 для успешного распознавания
            if best_score > 0.3:
                normalized_confidence = 0.6 + (best_score - 0.3) / 0.7 * 0.35
                normalized_confidence = min(0.95, max(0.6, normalized_confidence))
            else:
                normalized_confidence = best_score * 2.0  # Для очень низких значений
            
            return best_lang, normalized_confidence
            
        except Exception as e:
            # В случае ошибки возвращаем None
            return None, 0.0
    
    def detect_language(self, features: Dict, audio: np.ndarray = None, sr: float = None, audio_file: str = None) -> Tuple[str, float]:
        """
        Определяет язык по акустическим характеристикам голоса и SpeechRecognition
        Улучшенная версия с использованием SpeechRecognition и фильтрацией помех
        
        Args:
            features: Словарь с характеристиками
            audio: Аудио данные (опционально, для дополнительного анализа)
            sr: Частота дискретизации (опционально)
            audio_file: Путь к аудиофайлу (опционально, для более эффективного использования SpeechRecognition)
            
        Returns:
            Кортеж (язык, уверенность)
        """
        # Используем акустические характеристики для определения языка
        # Разные языки имеют разные паттерны в формантах, MFCC и спектральных характеристиках
        
        mfcc_mean = features.get('mfcc_mean', [])
        formant_f1 = features.get('formant_f1', 0)
        formant_f2 = features.get('formant_f2', 0)
        spectral_centroid = features.get('spectral_centroid_mean', 0)
        speech_rate = features.get('speech_rate', 0)
        pitch_median = features.get('pitch_median', 0)
        zcr_mean = features.get('zcr_mean', 0)
        rms_mean = features.get('rms_mean', 0)
        rms_std = features.get('rms_std', 0)
        
        if len(mfcc_mean) == 0:
            return ("Не определен", 0.0)
        
        # Проверка качества сигнала - обнаружение помех
        noise_penalty = 0.0
        
        # Высокий ZCR может указывать на шумы/помехи
        if zcr_mean > 0.15:  # Очень высокий ZCR - вероятны помехи
            noise_penalty += 0.3
        elif zcr_mean > 0.12:  # Высокий ZCR
            noise_penalty += 0.15
        
        # Высокая вариативность RMS может указывать на хлопки/всплески
        if rms_mean > 0 and rms_std / rms_mean > 1.5:  # Высокая относительная вариативность
            noise_penalty += 0.2
        
        # Необычные значения формант могут указывать на искажения
        if formant_f1 > 0 and formant_f2 > 0:
            f2_f1_ratio = formant_f2 / formant_f1
            # Очень высокое или очень низкое соотношение может быть из-за помех
            if f2_f1_ratio > 4.0 or f2_f1_ratio < 1.0:
                noise_penalty += 0.2
        
        # Характеристики для разных языков (на основе исследований акустики речи)
        # Это упрощенная эвристика, в реальности нужна обученная модель
        
        scores = {}
        
        # Русский язык - улучшенные и расширенные критерии с учетом новых исследований
        russian_score = 0.0
        russian_indicators = 0  # Счетчик индикаторов русского языка
        
        if formant_f1 > 0 and formant_f2 > 0:
            f2_f1_ratio = formant_f2 / formant_f1
            # Русский: средний диапазон формант, соотношение ~1.8-2.5 (оптимизировано)
            if 1.6 < f2_f1_ratio < 2.7:  # Расширен для лучшего покрытия различных диалектов
                russian_score += 0.55  # Увеличена важность
                russian_indicators += 1
                # Бонус за попадание в оптимальный диапазон
                if 1.8 < f2_f1_ratio < 2.4:
                    russian_score += 0.25
            # Русский: F1 обычно 380-720 Hz (расширенный диапазон для различных голосов)
            if 380 < formant_f1 < 720:
                russian_score += 0.35
                russian_indicators += 1
                # Бонус за оптимальный диапазон
                if 440 < formant_f1 < 660:
                    russian_score += 0.18
            # Русский: F2 обычно 1050-2100 Hz (расширенный диапазон)
            if 1050 < formant_f2 < 2100:
                russian_score += 0.35
                russian_indicators += 1
                # Бонус за оптимальный диапазон
                if 1150 < formant_f2 < 1850:
                    russian_score += 0.18
        
        # MFCC характеристики для русского (расширенные критерии с улучшенной точностью)
        if len(mfcc_mean) >= 3:
            mfcc_0 = mfcc_mean[0]
            mfcc_1 = mfcc_mean[1] if len(mfcc_mean) > 1 else 0
            mfcc_2 = mfcc_mean[2] if len(mfcc_mean) > 2 else 0
            mfcc_3 = mfcc_mean[3] if len(mfcc_mean) > 3 else 0
            mfcc_4 = mfcc_mean[4] if len(mfcc_mean) > 4 else 0
            
            # Русский: характерные значения MFCC (оптимизированные диапазоны)
            if -16 < mfcc_0 < -4:  # Расширенный диапазон
                russian_score += 0.28
                russian_indicators += 1
                if -13.5 < mfcc_0 < -6.5:  # Оптимальный диапазон
                    russian_score += 0.12
            if -6 < mfcc_1 < 6:  # Расширенный диапазон
                russian_score += 0.22
                russian_indicators += 1
                if -3 < mfcc_1 < 4:  # Оптимальный диапазон
                    russian_score += 0.1
            if -3 < mfcc_2 < 9:  # Расширенный диапазон
                russian_score += 0.18
                russian_indicators += 1
                if 0 < mfcc_2 < 7:  # Оптимальный диапазон
                    russian_score += 0.08
            # Дополнительная проверка по mfcc_3 и mfcc_4 для повышения точности
            if -4 < mfcc_3 < 8:
                russian_score += 0.12
                russian_indicators += 1
            if -2 < mfcc_4 < 6:
                russian_score += 0.08
        
        # Спектральный центроид для русского (обычно 1550-2450 Hz - расширенный диапазон)
        if 1550 < spectral_centroid < 2450:
            russian_score += 0.28
            russian_indicators += 1
            if 1650 < spectral_centroid < 2350:  # Оптимальный диапазон
                russian_score += 0.12
        
        # Дополнительные индикаторы русского языка (улучшенные)
        # Темп речи для русского обычно 3-7 слогов/сек (оптимизирован)
        if 2.3 < speech_rate < 8.0:  # Расширен для различных стилей речи
            russian_score += 0.18
            russian_indicators += 1
            if 2.8 < speech_rate < 7.0:  # Оптимальный диапазон
                russian_score += 0.12
        
        # ZCR для русского (обычно средние значения, оптимизировано)
        if 0.04 < zcr_mean < 0.13:
            russian_score += 0.12
            if 0.06 < zcr_mean < 0.11:  # Оптимальный диапазон
                russian_score += 0.06
        
        # Дополнительный анализ по pitch для русского языка
        if pitch_median > 0:
            # Русский язык имеет характерное распределение pitch
            if 80 < pitch_median < 280:  # Широкий диапазон для различных голосов
                russian_score += 0.1
                if 100 < pitch_median < 250:  # Оптимальный диапазон
                    russian_score += 0.05
        
        # Применяем штраф за помехи только если есть несколько индикаторов
        if russian_indicators >= 2:
            russian_score = max(0.0, russian_score - noise_penalty * 0.4)  # Меньший штраф
        elif russian_indicators >= 1:
            # Даже с одним индикатором даем базовый score
            russian_score = max(0.2, russian_score - noise_penalty * 0.3)
        
        scores['Русский'] = min(1.0, russian_score)
        
        # Английский язык - с учетом помех
        english_score = 0.0
        english_indicators = 0
        
        if formant_f1 > 0 and formant_f2 > 0:
            f2_f1_ratio = formant_f2 / formant_f1
            # Английский: более широкий диапазон соотношения формант
            # Но если есть помехи, это может быть ложным срабатыванием
            if 2.3 < f2_f1_ratio < 3.2:  # Более узкий диапазон для точности
                english_score += 0.3
                english_indicators += 1
            # Английский: F1 обычно 350-700 Hz (более узкий диапазон)
            if 350 < formant_f1 < 700:
                english_score += 0.15
            # Английский: F2 обычно 1500-2500 Hz (более узкий диапазон)
            if 1500 < formant_f2 < 2500:
                english_score += 0.15
        
        # MFCC для английского
        if len(mfcc_mean) >= 3:
            mfcc_0 = mfcc_mean[0]
            mfcc_1 = mfcc_mean[1] if len(mfcc_mean) > 1 else 0
            
            # Английский: другие значения MFCC
            if -11 < mfcc_0 < -5:  # Более узкий диапазон
                english_score += 0.15
            if 0 < mfcc_1 < 6:  # Более узкий диапазон
                english_score += 0.1
        
        # Спектральный центроид для английского (обычно 2000-2800 Hz - более узкий)
        if 2000 < spectral_centroid < 2800:
            english_score += 0.15
        
        # Штраф за помехи - если есть помехи, снижаем вероятность английского
        # (так как помехи могут искажать характеристики и делать их похожими на английский)
        if noise_penalty > 0.2 and english_indicators < 2:
            english_score = max(0.0, english_score - noise_penalty * 0.7)
        
        scores['Английский'] = min(1.0, english_score)
        
        # Испанский язык
        spanish_score = 0.0
        if formant_f1 > 0 and formant_f2 > 0:
            f2_f1_ratio = formant_f2 / formant_f1
            # Испанский: более высокие форманты
            if 1.5 < f2_f1_ratio < 2.2:
                spanish_score += 0.25
            if 450 < formant_f1 < 800:
                spanish_score += 0.15
            if 1100 < formant_f2 < 2400:
                spanish_score += 0.15
        
        if len(mfcc_mean) >= 2:
            mfcc_0 = mfcc_mean[0]
            if -14 < mfcc_0 < -4:
                spanish_score += 0.15
        
        if 1600 < spectral_centroid < 2800:
            spanish_score += 0.15
        
        scores['Испанский'] = min(1.0, spanish_score)
        
        # Немецкий язык (ужесточены критерии, чтобы не путать с русским)
        german_score = 0.0
        german_indicators = 0
        
        if formant_f1 > 0 and formant_f2 > 0:
            f2_f1_ratio = formant_f2 / formant_f1
            # Немецкий: характерные форманты (более строгие критерии)
            # Немецкий обычно имеет более низкое соотношение F2/F1, чем русский
            if 1.5 < f2_f1_ratio < 2.2:  # Более узкий диапазон, отличный от русского
                german_score += 0.3
                german_indicators += 1
            # Немецкий: F1 обычно ниже, чем у русского
            if 300 < formant_f1 < 600:  # Более низкий диапазон
                german_score += 0.2
                german_indicators += 1
            # Немецкий: F2 обычно ниже, чем у русского
            if 900 < formant_f2 < 1800:  # Более низкий диапазон
                german_score += 0.2
                german_indicators += 1
        
        if len(mfcc_mean) >= 2:
            mfcc_0 = mfcc_mean[0]
            # Немецкий обычно имеет более низкие значения MFCC
            if -18 < mfcc_0 < -8:  # Более низкий диапазон
                german_score += 0.2
                german_indicators += 1
        
        # Спектральный центроид для немецкого обычно ниже
        if 1300 < spectral_centroid < 2200:  # Более низкий диапазон
            german_score += 0.2
        
        # Штраф, если характеристики похожи на русский
        if russian_indicators >= 3 and german_indicators < 3:
            # Если русский имеет больше индикаторов, снижаем немецкий
            german_score *= 0.7
        
        scores['Немецкий'] = min(1.0, german_score)
        
        # Французский язык
        french_score = 0.0
        if formant_f1 > 0 and formant_f2 > 0:
            f2_f1_ratio = formant_f2 / formant_f1
            # Французский: более высокие форманты
            if 2.2 < f2_f1_ratio < 3.2:
                french_score += 0.25
            if 400 < formant_f1 < 750:
                french_score += 0.15
            if 1300 < formant_f2 < 2600:
                french_score += 0.15
        
        if len(mfcc_mean) >= 2:
            mfcc_0 = mfcc_mean[0]
            if -13 < mfcc_0 < -3:
                french_score += 0.15
        
        if 1700 < spectral_centroid < 2900:
            french_score += 0.15
        
        scores['Французский'] = min(1.0, french_score)
        
        # Находим язык с максимальным score (акустический метод) - улучшенная логика
        acoustic_language = None
        acoustic_confidence = 0.0
        
        if scores and max(scores.values()) >= 0.22:  # Оптимизирован порог для более точного определения
            acoustic_language = max(scores, key=scores.get)
            acoustic_score = scores[acoustic_language]
            
            # Проверяем, насколько результат лучше остальных
            sorted_scores = sorted(scores.values(), reverse=True)
            score_difference = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
            
            # Дополнительная проверка: отдаем предпочтение русскому, если он близок к другим языкам
            # Это важно, так как русский часто путают с немецким и другими языками
            russian_score_value = scores.get('Русский', 0)
            
            # Если русский имеет приличный score (выше 0.28), и текущий язык - не русский
            if acoustic_language != 'Русский' and russian_score_value > 0.28:
                # Проверяем разницу между текущим языком и русским
                score_diff = acoustic_score - russian_score_value
                
                # Если разница небольшая (менее 0.18), отдаем предпочтение русскому
                if score_diff < 0.18:
                    acoustic_language = 'Русский'
                    acoustic_score = russian_score_value
                    score_difference = acoustic_score - scores.get(acoustic_language, 0)
                # Если текущий язык - немецкий, и русский близок, отдаем предпочтение русскому
                elif acoustic_language == 'Немецкий' and russian_score_value > acoustic_score * 0.82:
                    acoustic_language = 'Русский'
                    acoustic_score = russian_score_value
                    score_difference = acoustic_score - scores.get('Немецкий', 0)
                # Если есть помехи, и русский близок - выбираем русский
                elif noise_penalty > 0.2 and russian_score_value > acoustic_score * 0.78:
                    acoustic_language = 'Русский'
                    acoustic_score = russian_score_value
                    score_difference = acoustic_score - scores.get(acoustic_language, 0)
            
            # Нормализуем уверенность с учетом разницы между лучшим и вторым результатом
            # Улучшенная формула с учетом качества данных
            base_confidence = 0.48 + (acoustic_score - 0.22) / 0.78 * 0.42
            
            # Бонус за большую разницу между результатами (улучшенная логика)
            if score_difference > 0.35:
                base_confidence += 0.12
            elif score_difference > 0.25:
                base_confidence += 0.08
            elif score_difference > 0.15:
                base_confidence += 0.04
            
            # Бонус за высокий общий score
            if acoustic_score > 0.6:
                base_confidence += 0.05
            
            acoustic_confidence = max(0.48, min(0.92, base_confidence))
            
            # Дополнительное снижение уверенности при наличии помех (улучшенная логика)
            if noise_penalty > 0.35:
                acoustic_confidence *= 0.82
            elif noise_penalty > 0.25:
                acoustic_confidence *= 0.90
            elif noise_penalty > 0.15:
                acoustic_confidence *= 0.95
        
        # Пробуем определить язык через SpeechRecognition
        speech_language = None
        speech_confidence = 0.0
        
        # Проверяем длительность аудио (SpeechRecognition лучше работает с аудио длиннее 0.5 сек)
        audio_duration = features.get('duration', 0)
        use_speech_recognition = audio_duration >= 0.3  # Минимум 0.3 секунды
        
        if use_speech_recognition:
            # Используем файл напрямую, если он доступен и в поддерживаемом формате
            use_file_directly = False
            if audio_file and os.path.exists(audio_file):
                # Проверяем, является ли файл в поддерживаемом формате
                # SpeechRecognition поддерживает WAV, AIFF, FLAC
                file_ext = os.path.splitext(audio_file)[1].lower()
                supported_formats = ['.wav', '.aiff', '.aif', '.flac']
                if file_ext in supported_formats:
                    use_file_directly = True
            
            if use_file_directly:
                # Используем файл напрямую (более эффективно)
                try:
                    speech_language, speech_confidence = self._detect_language_with_speech_recognition(audio_file)
                except Exception:
                    # В случае ошибки пробуем создать временный файл
                    use_file_directly = False
            
            if not use_file_directly and audio is not None and sr is not None:
                try:
                    # Сохраняем аудио во временный файл для SpeechRecognition
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                        # Нормализуем аудио для SpeechRecognition (нужен формат WAV, моно, 16-bit)
                        # Конвертируем в нужный формат
                        audio_normalized = audio.copy()
                        # Нормализуем амплитуду
                        if np.max(np.abs(audio_normalized)) > 0:
                            audio_normalized = audio_normalized / np.max(np.abs(audio_normalized))
                        # Конвертируем в int16
                        audio_int16 = (audio_normalized * 32767).astype(np.int16)
                        # Сохраняем как WAV
                        sf.write(temp_path, audio_int16, int(sr))
                        
                        # Определяем язык через SpeechRecognition
                        speech_language, speech_confidence = self._detect_language_with_speech_recognition(temp_path)
                        
                        # Удаляем временный файл
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                except Exception as e:
                    # В случае ошибки просто игнорируем SpeechRecognition
                    pass
        
        # Комбинируем результаты обоих методов (улучшенная логика)
        # SpeechRecognition имеет приоритет, так как он более точный
        # Особое внимание к русскому языку - отдаем ему предпочтение при неопределенности
        
        # Если SpeechRecognition определил русский, повышаем его приоритет
        if speech_language == 'Русский' and speech_confidence > 0.3:
            # Даже при низкой уверенности SpeechRecognition для русского даем ему шанс
            if acoustic_language != 'Русский' and acoustic_confidence < 0.6:
                # Если акустический метод не уверен, используем SpeechRecognition
                return (speech_language, max(0.55, speech_confidence))
        
        if speech_language and speech_confidence > 0.4:
            # Если SpeechRecognition дал результат (даже с умеренной уверенностью)
            if acoustic_language and acoustic_confidence > 0.4:
                # Если оба метода дали результат
                if speech_language == acoustic_language:
                    # Если методы согласны - значительно повышаем уверенность
                    final_confidence = min(0.95, (speech_confidence * 0.7 + acoustic_confidence * 0.3) + 0.15)
                    return (speech_language, final_confidence)
                else:
                    # Если методы не согласны, отдаем предпочтение SpeechRecognition
                    # но учитываем акустический метод для снижения уверенности
                    confidence_diff = abs(speech_confidence - acoustic_confidence)
                    if confidence_diff > 0.2:
                        # SpeechRecognition значительно лучше - используем его
                        return (speech_language, speech_confidence * 0.95)
                    else:
                        # Методы близки - используем SpeechRecognition, но снижаем уверенность
                        return (speech_language, speech_confidence * 0.85)
            else:
                # Только SpeechRecognition дал результат - используем его
                return (speech_language, speech_confidence)
        elif speech_language and speech_confidence > 0.3:
            # SpeechRecognition дал результат с низкой уверенностью, но все равно используем его
            # если акустический метод не дал лучшего результата
            if acoustic_language and acoustic_confidence > speech_confidence + 0.2:
                # Акустический метод значительно лучше
                return (acoustic_language, acoustic_confidence)
            else:
                # Используем SpeechRecognition, но снижаем уверенность
                return (speech_language, max(0.5, speech_confidence * 0.9))
        elif acoustic_language and acoustic_confidence > 0.4:
            # Только акустический метод дал хороший результат
            return (acoustic_language, acoustic_confidence)
        elif speech_language:
            # SpeechRecognition дал результат, но с очень низкой уверенностью
            # Используем его, если акустический метод не лучше
            if acoustic_language and acoustic_confidence > speech_confidence:
                return (acoustic_language, acoustic_confidence)
            else:
                return (speech_language, max(0.4, speech_confidence))
        elif acoustic_language:
            # Только акустический метод дал результат
            return (acoustic_language, acoustic_confidence)
        else:
            # Ни один метод не дал результата
            return ("Не определен", 0.0)
    
    def analyze_voice(self, file_path: str) -> Dict:
        """
        Полный анализ голоса
        
        Args:
            file_path: Путь к аудиофайлу
            
        Returns:
            Словарь с результатами анализа
        """
        # Загружаем аудио
        audio, sr = self.load_audio(file_path)
        
        # Извлекаем характеристики
        features = self.extract_features(audio, sr)
        
        # Определяем пол
        gender, gender_confidence = self.detect_gender(features)
        
        # Определяем язык (передаем также путь к файлу для более эффективного использования SpeechRecognition)
        # Определяем язык ПЕРЕД акцентом, чтобы использовать информацию о языке
        language, language_confidence = self.detect_language(features, audio, sr, file_path)
        
        # Определяем акцент (передаем язык для более точного определения)
        accent, accent_confidence = self.detect_accent(features, language)
        
        # Определяем эмоции
        emotion, emotion_confidence = self.detect_emotion(features)
        
        # Формируем результат
        result = {
            'audio': audio,
            'sample_rate': sr,
            'features': features,
            'gender': gender,
            'gender_confidence': gender_confidence,
            'accent': accent,
            'accent_confidence': accent_confidence,
            'emotion': emotion,
            'emotion_confidence': emotion_confidence,
            'language': language,
            'language_confidence': language_confidence,
            'duration': features.get('duration', 0)
        }
        
        return result

