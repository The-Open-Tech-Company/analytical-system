"""
Analytical System - Voice Comparator Module
Voice track comparison

Repository: https://github.com/The-Open-Tech-Company/analytical-system
License: Unlicense (Open Source)
"""
import numpy as np
import hashlib
import os
from typing import Dict, Tuple
from voice_analyzer import VoiceAnalyzer


class VoiceComparator:
    """Класс для сравнения двух голосовых треков"""
    
    def __init__(self):
        """Инициализация компаратора голосов"""
        self.analyzer = VoiceAnalyzer()
    
    def compare_features(self, features1: Dict, features2: Dict) -> Dict[str, float]:
        """
        Сравнивает характеристики двух голосов
        
        Args:
            features1: Характеристики первого голоса
            features2: Характеристики второго голоса
            
        Returns:
            Словарь с процентами совпадения для каждой характеристики
        """
        results = {}
        
        # Сравнение основной частоты (pitch)
        pitch1 = features1.get('pitch_mean', 0)
        pitch2 = features2.get('pitch_mean', 0)
        if pitch1 > 0 and pitch2 > 0:
            pitch_similarity = self._compare_values(pitch1, pitch2, tolerance=0.2)
            results['pitch'] = pitch_similarity
        else:
            results['pitch'] = 0.0
        
        # Сравнение вариативности pitch
        pitch_std1 = features1.get('pitch_std', 0)
        pitch_std2 = features2.get('pitch_std', 0)
        if pitch_std1 > 0 and pitch_std2 > 0:
            pitch_std_similarity = self._compare_values(pitch_std1, pitch_std2, tolerance=0.3)
            results['pitch_variation'] = pitch_std_similarity
        else:
            results['pitch_variation'] = 0.0
        
        # Сравнение MFCC
        mfcc1 = features1.get('mfcc_mean', [])
        mfcc2 = features2.get('mfcc_mean', [])
        if len(mfcc1) > 0 and len(mfcc2) > 0:
            mfcc_similarity = self._compare_vectors(mfcc1, mfcc2)
            results['mfcc'] = mfcc_similarity
        else:
            results['mfcc'] = 0.0
        
        # Сравнение спектрального центроида
        sc1 = features1.get('spectral_centroid_mean', 0)
        sc2 = features2.get('spectral_centroid_mean', 0)
        if sc1 > 0 and sc2 > 0:
            sc_similarity = self._compare_values(sc1, sc2, tolerance=0.25)
            results['spectral_centroid'] = sc_similarity
        else:
            results['spectral_centroid'] = 0.0
        
        # Сравнение формант
        f1_1 = features1.get('formant_f1', 0)
        f1_2 = features2.get('formant_f1', 0)
        f2_1 = features1.get('formant_f2', 0)
        f2_2 = features2.get('formant_f2', 0)
        
        if f1_1 > 0 and f1_2 > 0:
            f1_similarity = self._compare_values(f1_1, f1_2, tolerance=0.3)
            results['formant_f1'] = f1_similarity
        else:
            results['formant_f1'] = 0.0
        
        if f2_1 > 0 and f2_2 > 0:
            f2_similarity = self._compare_values(f2_1, f2_2, tolerance=0.3)
            results['formant_f2'] = f2_similarity
        else:
            results['formant_f2'] = 0.0
        
        # Сравнение энергии (RMS)
        rms1 = features1.get('rms_mean', 0)
        rms2 = features2.get('rms_mean', 0)
        if rms1 > 0 and rms2 > 0:
            rms_similarity = self._compare_values(rms1, rms2, tolerance=0.3)
            results['energy'] = rms_similarity
        else:
            results['energy'] = 0.0
        
        # Сравнение zero crossing rate
        zcr1 = features1.get('zcr_mean', 0)
        zcr2 = features2.get('zcr_mean', 0)
        if zcr1 > 0 and zcr2 > 0:
            zcr_similarity = self._compare_values(zcr1, zcr2, tolerance=0.3)
            results['zcr'] = zcr_similarity
        else:
            results['zcr'] = 0.0
        
        # Сравнение темпа речи
        rate1 = features1.get('speech_rate', 0)
        rate2 = features2.get('speech_rate', 0)
        if rate1 > 0 and rate2 > 0:
            rate_similarity = self._compare_values(rate1, rate2, tolerance=0.3)
            results['speech_rate'] = rate_similarity
        else:
            results['speech_rate'] = 0.0
        
        # Вычисляем общее совпадение
        valid_results = [v for v in results.values() if v > 0 and isinstance(v, (int, float))]
        
        # Проверяем, не являются ли все характеристики очень близкими (почти идентичные файлы)
        high_similarity_count = sum(1 for v in valid_results if v > 90)
        if high_similarity_count >= len(valid_results) * 0.8 and len(valid_results) >= 5:
            # Если 80% характеристик имеют >90% совпадение, это почти идентичные файлы
            # Повышаем общее совпадение
            avg_similarity = np.mean(valid_results)
            if avg_similarity > 85:
                results['overall'] = min(100.0, avg_similarity * 1.1)
            else:
                results['overall'] = avg_similarity
        elif valid_results:
            # Взвешенное среднее (MFCC и pitch имеют больший вес)
            weights = {
                'pitch': 0.25,
                'pitch_variation': 0.10,
                'mfcc': 0.30,
                'spectral_centroid': 0.10,
                'formant_f1': 0.08,
                'formant_f2': 0.08,
                'energy': 0.04,
                'zcr': 0.03,
                'speech_rate': 0.02
            }
            
            weighted_sum = 0.0
            total_weight = 0.0
            for key, value in results.items():
                if value > 0 and key in weights and isinstance(value, (int, float)):
                    weighted_sum += value * weights[key]
                    total_weight += weights[key]
            
            if total_weight > 0:
                overall = weighted_sum / total_weight
                # Если среднее взвешенное > 85%, немного повышаем для учета очень похожих голосов
                if overall > 85:
                    overall = min(100.0, overall * 1.05)
                results['overall'] = overall
            else:
                results['overall'] = np.mean(valid_results)
        else:
            results['overall'] = 0.0
        
        return results
    
    def _compare_values(self, val1: float, val2: float, tolerance: float = 0.2) -> float:
        """
        Сравнивает два значения и возвращает процент совпадения (улучшенная версия)
        
        Args:
            val1: Первое значение
            val2: Второе значение
            tolerance: Допустимое отклонение (0.2 = 20%)
            
        Returns:
            Процент совпадения (0-100)
        """
        if val1 == 0 and val2 == 0:
            return 100.0
        
        if val1 == 0 or val2 == 0:
            return 0.0
        
        # Вычисляем относительную разницу
        diff = abs(val1 - val2)
        avg = (val1 + val2) / 2.0
        
        if avg == 0:
            return 100.0 if diff == 0 else 0.0
        
        relative_diff = diff / avg
        
        # Если значения идентичны или очень близки (относительная разница < 1%)
        if relative_diff < 0.01:
            return 100.0
        # Если значения очень близки (относительная разница < tolerance)
        elif relative_diff < tolerance:
            # Линейная интерполяция от 100% до 90% в диапазоне 0.01 - tolerance
            similarity = 100.0 - (relative_diff - 0.01) / (tolerance - 0.01) * 10.0
            return max(90.0, min(100.0, similarity))
        else:
            # Экспоненциальное затухание для больших различий
            # Используем более мягкую функцию для лучшего распознавания похожих значений
            ratio = min(val1, val2) / max(val1, val2)
            # Более мягкая функция затухания
            similarity = 100.0 * (ratio ** 0.5)
            return max(0.0, min(100.0, similarity))
    
    def _compare_vectors(self, vec1: list, vec2: list) -> float:
        """
        Сравнивает два вектора и возвращает процент совпадения (улучшенная версия)
        
        Args:
            vec1: Первый вектор
            vec2: Второй вектор
            
        Returns:
            Процент совпадения (0-100)
        """
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        # Приводим к одинаковой длине
        min_len = min(len(vec1), len(vec2))
        vec1 = np.array(vec1[:min_len])
        vec2 = np.array(vec2[:min_len])
        
        # Проверяем на идентичность (для идентичных файлов)
        if np.allclose(vec1, vec2, rtol=1e-5, atol=1e-8):
            return 100.0
        
        # Метод 1: Косинусное сходство
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        cosine_similarity = 0.0
        if norm1 > 0 and norm2 > 0:
            cosine_similarity = dot_product / (norm1 * norm2)
        
        # Метод 2: Евклидово расстояние (нормализованное)
        euclidean_dist = np.linalg.norm(vec1 - vec2)
        avg_norm = (norm1 + norm2) / 2.0
        euclidean_similarity = 0.0
        if avg_norm > 0:
            # Нормализуем расстояние относительно среднего размера векторов
            normalized_dist = euclidean_dist / avg_norm
            # Преобразуем в сходство (чем меньше расстояние, тем больше сходство)
            euclidean_similarity = max(0.0, 1.0 - normalized_dist)
        
        # Метод 3: Корреляция Пирсона
        if len(vec1) > 1:
            correlation = np.corrcoef(vec1, vec2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 1.0 if vec1[0] == vec2[0] else 0.0
        
        # Комбинируем результаты (взвешенное среднее)
        # Косинусное сходство - основной метод
        cosine_weight = 0.5
        euclidean_weight = 0.3
        correlation_weight = 0.2
        
        # Преобразуем косинусное сходство в проценты (от -1 до 1 -> от 0 до 100)
        cosine_percent = (cosine_similarity + 1) / 2 * 100
        euclidean_percent = euclidean_similarity * 100
        correlation_percent = (correlation + 1) / 2 * 100
        
        # Комбинируем
        combined_similarity = (cosine_percent * cosine_weight + 
                              euclidean_percent * euclidean_weight + 
                              correlation_percent * correlation_weight)
        
        # Если все методы показывают высокое сходство (>90%), повышаем результат
        if cosine_percent > 90 and euclidean_percent > 90 and correlation_percent > 90:
            combined_similarity = min(100.0, combined_similarity * 1.1)
        
        return max(0.0, min(100.0, combined_similarity))
    
    def _file_hash(self, file_path: str) -> str:
        """Вычисляет MD5 хеш файла"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def compare_voices(self, file_path1: str, file_path2: str) -> Dict:
        """
        Сравнивает два голосовых трека
        
        Args:
            file_path1: Путь к первому аудиофайлу
            file_path2: Путь ко второму аудиофайлу
            
        Returns:
            Словарь с результатами сравнения
        """
        # Проверяем, не являются ли файлы идентичными (по пути или по содержимому)
        if file_path1 == file_path2:
            # Если это один и тот же файл, возвращаем 100% совпадение
            analysis1 = self.analyzer.analyze_voice(file_path1)
            comparison_results = {}
            for key in ['pitch', 'pitch_variation', 'mfcc', 'spectral_centroid', 
                       'formant_f1', 'formant_f2', 'energy', 'zcr', 'speech_rate']:
                comparison_results[key] = 100.0
            comparison_results['overall'] = 100.0
            
            return {
                'voice1': analysis1,
                'voice2': analysis1,
                'comparison': comparison_results
            }
        
        # Проверяем идентичность по содержимому (хеш)
        if os.path.exists(file_path1) and os.path.exists(file_path2):
            hash1 = self._file_hash(file_path1)
            hash2 = self._file_hash(file_path2)
            if hash1 and hash2 and hash1 == hash2:
                # Файлы идентичны по содержимому
                analysis1 = self.analyzer.analyze_voice(file_path1)
                comparison_results = {}
                for key in ['pitch', 'pitch_variation', 'mfcc', 'spectral_centroid', 
                           'formant_f1', 'formant_f2', 'energy', 'zcr', 'speech_rate']:
                    comparison_results[key] = 100.0
                comparison_results['overall'] = 100.0
                
                return {
                    'voice1': analysis1,
                    'voice2': analysis1,
                    'comparison': comparison_results
                }
        
        # Анализируем оба голоса
        analysis1 = self.analyzer.analyze_voice(file_path1)
        analysis2 = self.analyzer.analyze_voice(file_path2)
        
        # Сравниваем характеристики
        comparison_results = self.compare_features(
            analysis1['features'],
            analysis2['features']
        )
        
        # Формируем полный результат
        result = {
            'voice1': analysis1,
            'voice2': analysis2,
            'comparison': comparison_results
        }
        
        return result

