"""
Analytical System - Face Comparator Module
Face comparison and similarity calculation

Repository: https://github.com/The-Open-Tech-Company/analytical-system
License: Unlicense (Open Source)
"""
import numpy as np
from typing import Dict, List, Tuple
from face_analyzer import FaceAnalyzer


class FaceComparator:
    """Класс для сравнения двух лиц"""
    
    def __init__(self):
        self.analyzer = FaceAnalyzer()
    
    def normalize_points(self, points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
        """
        Нормализует точки относительно референсных точек для сравнения
        
        Args:
            points: Точки для нормализации
            reference_points: Референсные точки (например, овал лица)
            
        Returns:
            Нормализованные точки
        """
        if len(points) == 0 or len(reference_points) == 0:
            return points
        
        # Вычисляем центроид и масштаб референсных точек
        ref_centroid = np.mean(reference_points, axis=0)
        ref_distances = [np.linalg.norm(p - ref_centroid) for p in reference_points]
        ref_scale = np.mean(ref_distances) if ref_distances else 1.0
        
        # Нормализуем точки
        points_centroid = np.mean(points, axis=0)
        normalized = (points - points_centroid) / ref_scale if ref_scale > 0 else points
        
        return normalized
    
    def compare_points(self, points1: np.ndarray, points2: np.ndarray, 
                      normalize: bool = True, reference1: np.ndarray = None,
                      reference2: np.ndarray = None) -> float:
        """
        Сравнивает два набора точек и возвращает процент совпадения
        Улучшен для работы с неточными данными (плохие фото)
        
        Args:
            points1: Первый набор точек
            points2: Второй набор точек
            normalize: Нужно ли нормализовать точки
            reference1: Референсные точки для нормализации первого набора
            reference2: Референсные точки для нормализации второго набора
            
        Returns:
            Процент совпадения (0-100)
        """
        if len(points1) == 0 or len(points2) == 0:
            return 0.0
        
        # Нормализуем точки, если нужно
        if normalize:
            if reference1 is not None and len(reference1) > 0:
                points1 = self.normalize_points(points1, reference1)
            if reference2 is not None and len(reference2) > 0:
                points2 = self.normalize_points(points2, reference2)
        
        # Приводим к одинаковому количеству точек (интерполяция)
        if len(points1) != len(points2):
            points1, points2 = self._resample_points(points1, points2)
        
        # Вычисляем расстояния между соответствующими точками (векторизованная операция)
        diff = points1 - points2
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        
        if len(distances) == 0:
            return 0.0
        
        # Устойчивая метрика: комбинация медианы и среднего
        median_distance = float(np.median(distances))
        mean_distance = float(np.mean(distances))
        robust_distance = float(0.6 * median_distance + 0.4 * mean_distance)
        
        # Вычисляем масштаб на основе размера характеристик
        size1 = np.max(points1, axis=0) - np.min(points1, axis=0)
        size2 = np.max(points2, axis=0) - np.min(points2, axis=0)
        avg_size = float((np.mean(size1) + np.mean(size2)) / 2.0)
        
        # Нормализуем расстояние относительно размера характеристики
        if avg_size > 0:
            normalized_distance = float(robust_distance / avg_size)
        else:
            normalized_distance = float(robust_distance)
        
        # Строгая функция преобразования расстояния в сходство
        similarity = float(np.exp(-normalized_distance * 12.0))
        
        # Преобразуем в проценты и ограничиваем диапазон
        percentage = float(max(0, min(100, similarity * 100)))
        
        return percentage
    
    def _resample_points(self, points1: np.ndarray, points2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Приводит два набора точек к одинаковому количеству точек"""
        target_count = max(len(points1), len(points2))
        
        if len(points1) < target_count:
            points1 = self._interpolate_points(points1, target_count)
        if len(points2) < target_count:
            points2 = self._interpolate_points(points2, target_count)
        
        return points1, points2
    
    def _interpolate_points(self, points: np.ndarray, target_count: int) -> np.ndarray:
        """Интерполирует точки для получения нужного количества"""
        if len(points) == 0:
            return points
        
        if len(points) >= target_count:
            # Прореживаем
            indices = np.linspace(0, len(points) - 1, target_count, dtype=np.int32)
            return points[indices]
        else:
            # Добавляем точки через интерполяцию
            new_points = []
            for i in range(target_count):
                pos = i / (target_count - 1) * (len(points) - 1)
                idx = int(pos)
                frac = pos - idx
                
                if idx < len(points) - 1:
                    p1 = points[idx]
                    p2 = points[idx + 1]
                    new_point = p1 + (p2 - p1) * frac
                    new_points.append(new_point)
                else:
                    new_points.append(points[-1])
            
            return np.array(new_points)
    
    def compare_metrics(self, metrics1: Dict, metrics2: Dict) -> float:
        """
        Сравнивает метрики двух характеристик
        Улучшен для работы с неточными данными
        
        Args:
            metrics1: Метрики первой характеристики
            metrics2: Метрики второй характеристики
            
        Returns:
            Процент совпадения (0-100)
        """
        if not metrics1 or not metrics2:
            return 0.0
        
        similarities = []
        weights = []
        
        # Сравниваем площадь с строгой метрикой
        if 'area' in metrics1 and 'area' in metrics2:
            area1 = metrics1['area']
            area2 = metrics2['area']
            if area1 > 0 and area2 > 0:
                ratio = min(area1, area2) / max(area1, area2)
                area_sim = ratio * ratio
                similarities.append(area_sim)
                weights.append(0.25)
        
        # Сравниваем соотношение сторон
        if 'aspect_ratio' in metrics1 and 'aspect_ratio' in metrics2:
            ar1 = metrics1['aspect_ratio']
            ar2 = metrics2['aspect_ratio']
            if ar1 > 0 and ar2 > 0:
                ratio = min(ar1, ar2) / max(ar1, ar2)
                ar_sim = ratio * ratio
                similarities.append(ar_sim)
                weights.append(0.20)
        
        # Сравниваем расстояния от центроида (более устойчивая метрика)
        if 'distances' in metrics1 and 'distances' in metrics2:
            dist1 = metrics1['distances']
            dist2 = metrics2['distances']
            if len(dist1) > 0 and len(dist2) > 0:
                min_len = min(len(dist1), len(dist2))
                dist1_norm = np.array(dist1[:min_len])
                dist2_norm = np.array(dist2[:min_len])
                
                # Нормализуем
                max_dist = max(np.max(dist1_norm), np.max(dist2_norm))
                if max_dist > 0:
                    dist1_norm = dist1_norm / max_dist
                    dist2_norm = dist2_norm / max_dist
                    
                    # Используем медиану для более устойчивой оценки
                    diff = np.abs(dist1_norm - dist2_norm)
                    median_diff = float(np.median(diff))
                    mean_diff = float(np.mean(diff))
                    # Комбинация медианы и среднего
                    robust_diff = float(0.6 * median_diff + 0.4 * mean_diff)
                    dist_sim = float(1.0 - robust_diff)
                    similarities.append(float(max(0, dist_sim)))
                    weights.append(0.30)
        
        # Сравниваем углы (более устойчивая метрика)
        if 'angles' in metrics1 and 'angles' in metrics2:
            angles1 = metrics1['angles']
            angles2 = metrics2['angles']
            if len(angles1) > 0 and len(angles2) > 0:
                min_len = min(len(angles1), len(angles2))
                angles1_norm = np.array(angles1[:min_len])
                angles2_norm = np.array(angles2[:min_len])
                
                # Нормализуем углы с учетом периодичности
                angle_diff = np.abs(angles1_norm - angles2_norm)
                # Учитываем периодичность углов (2π)
                angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
                
                # Используем медиану для более устойчивой оценки
                median_diff = float(np.median(angle_diff))
                mean_diff = float(np.mean(angle_diff))
                robust_diff = float(0.6 * median_diff + 0.4 * mean_diff)
                
                angle_sim = float(1.0 - (robust_diff / np.pi))
                similarities.append(float(max(0, angle_sim)))
                weights.append(0.25)
        
        if len(similarities) == 0:
            return 0.0
        
        # Взвешенное среднее значение всех сходств
        if len(weights) == len(similarities):
            # Нормализуем веса
            total_weight = float(sum(weights))
            if total_weight > 0:
                weights = [float(w / total_weight) for w in weights]
                overall_similarity = float(sum(float(s) * float(w) for s, w in zip(similarities, weights)))
            else:
                overall_similarity = float(np.mean(similarities))
        else:
            overall_similarity = float(np.mean(similarities))
        
        return float(max(0, min(100, overall_similarity * 100)))
    
    def normalize_face_size(self, features1: Dict, features2: Dict) -> Tuple[Dict, Dict]:
        """
        Нормализует размеры лиц перед сравнением, подгоняя их под один размер
        
        Args:
            features1: Характеристики первого лица
            features2: Характеристики второго лица
            
        Returns:
            Кортеж (нормализованные характеристики первого лица, нормализованные характеристики второго лица)
        """
        normalized_features1 = features1.copy()
        normalized_features2 = features2.copy()
        
        # Получаем овалы лиц для определения размера
        face_oval1 = features1.get('face_oval', np.array([]))
        face_oval2 = features2.get('face_oval', np.array([]))
        
        if len(face_oval1) == 0 or len(face_oval2) == 0:
            return normalized_features1, normalized_features2
        
        # Вычисляем размеры лиц (средний размер овала)
        size1 = np.max(face_oval1, axis=0) - np.min(face_oval1, axis=0)
        size2 = np.max(face_oval2, axis=0) - np.min(face_oval2, axis=0)
        
        avg_size1 = float(np.mean(size1))
        avg_size2 = float(np.mean(size2))
        
        # Выбираем целевой размер (средний между двумя)
        target_size = float((avg_size1 + avg_size2) / 2.0)
        
        if target_size <= 0:
            return normalized_features1, normalized_features2
        
        # Вычисляем масштабы для нормализации
        scale1 = float(target_size / avg_size1) if avg_size1 > 0 else 1.0
        scale2 = float(target_size / avg_size2) if avg_size2 > 0 else 1.0
        
        # Вычисляем центроиды для центрирования
        centroid1 = np.mean(face_oval1, axis=0)
        centroid2 = np.mean(face_oval2, axis=0)
        
        # Нормализуем все характеристики
        for feature_name in normalized_features1.keys():
            if feature_name in ['image', 'landmarks_2d', 'all_landmarks', 'gender', 'age', 'race', 
                               'gender_confidence', 'age_confidence', 'race_confidence', 'confidence']:
                continue
            
            points1 = normalized_features1.get(feature_name, np.array([]))
            points2 = normalized_features2.get(feature_name, np.array([]))
            
            if len(points1) > 0:
                # Масштабируем относительно центроида первого лица
                scaled_points1 = (points1 - centroid1) * scale1 + centroid1
                normalized_features1[feature_name] = scaled_points1
            
            if len(points2) > 0:
                # Масштабируем относительно центроида второго лица
                scaled_points2 = (points2 - centroid2) * scale2 + centroid2
                normalized_features2[feature_name] = scaled_points2
        
        return normalized_features1, normalized_features2
    
    def align_features_by_rotation(self, features: Dict, angle: float) -> Dict:
        """
        Поворачивает характеристики лица на указанный угол
        
        Args:
            features: Характеристики лица
            angle: Угол поворота в градусах
            
        Returns:
            Повернутые характеристики
        """
        if abs(angle) < 0.1:
            return features
        
        # Получаем центроид овала лица для поворота
        face_oval = features.get('face_oval', np.array([]))
        if len(face_oval) == 0:
            return features
        
        centroid = np.mean(face_oval, axis=0)
        
        # Создаем матрицу поворота
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Поворачиваем все характеристики
        aligned_features = features.copy()
        for feature_name in aligned_features.keys():
            if feature_name in ['image', 'landmarks_2d', 'all_landmarks', 'gender', 'age', 'race', 
                               'gender_confidence', 'age_confidence', 'race_confidence', 'confidence']:
                continue
            
            points = aligned_features.get(feature_name, np.array([]))
            if len(points) > 0:
                # Смещаем к началу координат, поворачиваем, возвращаем обратно
                centered_points = points - centroid
                rotated_points = (rotation_matrix @ centered_points.T).T
                aligned_features[feature_name] = rotated_points + centroid
        
        return aligned_features
    
    def _compare_faces_internal(self, features1: Dict, features2: Dict) -> Dict[str, float]:
        """
        Внутренний метод для сравнения двух лиц без автоматического выравнивания
        Используется для избежания рекурсии
        УЛУЧШЕНО для идентификации личности (полицейское использование)
        
        Args:
            features1: Характеристики первого лица
            features2: Характеристики второго лица
            
        Returns:
            Словарь с процентами совпадения для каждой характеристики
        """
        results = {}
        
        # Референсные точки для нормализации (овал лица)
        ref1 = features1.get('face_oval', np.array([]))
        ref2 = features2.get('face_oval', np.array([]))
        
        # Критические характеристики для идентификации личности (имеют больший вес)
        critical_features = ['left_eye', 'right_eye', 'nose_tip', 'nose_bridge', 
                           'mouth_outer', 'face_oval', 'chin']
        
        # Список характеристик для сравнения (добавлены характеристики волос)
        feature_names = [
            'face_oval', 'head_shape', 'left_eye', 'right_eye',
            'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip',
            'nose_contour', 'mouth_outer', 'mouth_inner', 'upper_lip',
            'lower_lip', 'left_cheek', 'right_cheek', 'left_ear',
            'right_ear', 'left_ear_detail', 'right_ear_detail', 'chin', 'forehead',
            'hair', 'hairline', 'left_temple', 'right_temple'  # Новые характеристики для волос
        ]
        
        critical_similarities = []
        
        for feature_name in feature_names:
            points1 = features1.get(feature_name, np.array([]))
            points2 = features2.get(feature_name, np.array([]))
            
            if len(points1) > 0 and len(points2) > 0:
                # Сравниваем точки
                point_similarity = self.compare_points(
                    points1, points2, 
                    normalize=True, 
                    reference1=ref1, 
                    reference2=ref2
                )
                
                # Вычисляем метрики
                metrics1 = self.analyzer.calculate_feature_metrics(points1)
                metrics2 = self.analyzer.calculate_feature_metrics(points2)
                
                # Сравниваем метрики
                metrics_similarity = self.compare_metrics(metrics1, metrics2)
                
                # Комбинируем результаты (среднее взвешенное)
                # Для плохих фото увеличиваем вес метрик, которые более устойчивы
                combined_similarity = (point_similarity * 0.65 + metrics_similarity * 0.35)
                
                results[feature_name] = min(100, combined_similarity)
                
                # Сохраняем совпадения критических характеристик
                if feature_name in critical_features:
                    critical_similarities.append(combined_similarity)
            else:
                results[feature_name] = 0.0
        
        # Вычисляем общий процент совпадения с учетом критических характеристик
        valid_results = [v for v in results.values() if v > 0]
        if valid_results:
            # Базовое среднее всех характеристик
            base_similarity = np.mean(valid_results)
            
            # Если есть критические характеристики, используем взвешенное среднее
            if critical_similarities:
                # Критические характеристики имеют вес 60%, остальные - 40%
                critical_avg = np.mean(critical_similarities)
                non_critical = [v for k, v in results.items() 
                               if k not in critical_features and v > 0]
                if non_critical:
                    non_critical_avg = np.mean(non_critical)
                    # Взвешенное среднее: критические 60%, остальные 40%
                    results['overall'] = critical_avg * 0.6 + non_critical_avg * 0.4
                else:
                    results['overall'] = critical_avg
            else:
                results['overall'] = base_similarity
            
            # Снижение общего совпадения при различиях в критических характеристиках
            if critical_similarities:
                min_critical = min(critical_similarities)
                if min_critical < 50:
                    results['overall'] = results['overall'] * 0.5
                elif min_critical < 60:
                    results['overall'] = results['overall'] * 0.75
                elif min_critical < 70:
                    results['overall'] = results['overall'] * 0.9
        else:
            results['overall'] = 0.0
        
        return results
    
    def compare_faces_with_rotation(self, features1: Dict, features2: Dict, 
                                   max_rotation: float = 15.0, step: float = 2.0) -> Tuple[Dict[str, float], float]:
        """
        Сравнивает два лица, пробуя различные углы поворота второго лица
        
        Args:
            features1: Характеристики первого лица (эталон)
            features2: Характеристики второго лица (поворачиваем)
            max_rotation: Максимальный угол поворота в градусах
            step: Шаг поворота в градусах
            
        Returns:
            Кортеж (лучшие результаты сравнения, оптимальный угол поворота)
        """
        best_results = None
        best_angle = 0.0
        best_score = 0.0
        
        # Пробуем разные углы поворота
        angles_to_try = [0.0]  # Начинаем с 0
        for angle in np.arange(-max_rotation, max_rotation + step, step):
            if abs(angle) > 0.1:
                angles_to_try.append(angle)
        
        for angle in angles_to_try:
            # Поворачиваем второе лицо
            if abs(angle) > 0.1:
                rotated_features2 = self.align_features_by_rotation(features2.copy(), angle)
            else:
                rotated_features2 = features2.copy()
            
            # Сравниваем используя внутренний метод (без рекурсии)
            results = self._compare_faces_internal(features1, rotated_features2)
            overall_score = results.get('overall', 0.0)
            
            # Сохраняем лучший результат
            if overall_score > best_score:
                best_score = overall_score
                best_results = results
                best_angle = angle
        
        return best_results, best_angle
    
    def compare_faces(self, features1: Dict, features2: Dict) -> Dict[str, float]:
        """
        Сравнивает два лица по всем характеристикам
        Автоматически выравнивает лица перед сравнением для устойчивости к наклонам
        УЛУЧШЕНО для идентификации личности (полицейское использование)
        
        Args:
            features1: Характеристики первого лица
            features2: Характеристики второго лица
            
        Returns:
            Словарь с процентами совпадения для каждой характеристики
        """
        # Проверка: если пол разный с высокой уверенностью, это разные люди
        gender1 = features1.get('gender', 'Не определен')
        gender2 = features2.get('gender', 'Не определен')
        gender_conf1 = features1.get('gender_confidence', 0.0)
        gender_conf2 = features2.get('gender_confidence', 0.0)
        
        if (gender1 != 'Не определен' and gender2 != 'Не определен' and 
            gender1 != gender2 and gender_conf1 > 0.7 and gender_conf2 > 0.7):
            results = {}
            for feature_name in ['face_oval', 'head_shape', 'left_eye', 'right_eye',
                                'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip',
                                'nose_contour', 'mouth_outer', 'mouth_inner', 'upper_lip',
                                'lower_lip', 'left_cheek', 'right_cheek', 'left_ear',
                                'right_ear', 'left_ear_detail', 'right_ear_detail', 'chin', 
                                'forehead', 'hair', 'hairline', 'left_temple', 'right_temple']:
                results[feature_name] = 0.0
            results['overall'] = 0.0
            return results
        
        # Автоматически выравниваем лица по углу наклона
        # Вычисляем углы наклона для обоих лиц
        angle1 = self.analyzer.calculate_face_angle(features1)
        angle2 = self.analyzer.calculate_face_angle(features2)
        
        # Выравниваем оба лица (поворачиваем к горизонтали)
        if abs(angle1) > 0.5:
            features1 = self.align_features_by_rotation(features1, -angle1)
        if abs(angle2) > 0.5:
            features2 = self.align_features_by_rotation(features2, -angle2)
        
        # Нормализуем размеры лиц перед сравнением
        features1, features2 = self.normalize_face_size(features1, features2)
        
        # Пробуем сравнение с дополнительными небольшими поворотами для точности
        # Это помогает найти оптимальное выравнивание
        results, optimal_angle = self.compare_faces_with_rotation(
            features1, features2, 
            max_rotation=10.0,  # Пробуем повороты до 10 градусов
            step=1.0  # Шаг 1 градус
        )
        
        # Дополнительная проверка: снижение совпадения при различии пола
        if (gender1 != 'Не определен' and gender2 != 'Не определен' and 
            gender1 != gender2):
            avg_gender_conf = (gender_conf1 + gender_conf2) / 2.0
            if avg_gender_conf > 0.5:
                results['overall'] = results.get('overall', 0.0) * 0.3
            elif avg_gender_conf > 0.3:
                results['overall'] = results.get('overall', 0.0) * 0.6
        
        return results
    
    def compare_faces_detailed(self, features1: Dict, features2: Dict) -> Dict[str, Dict]:
        """
        Сравнивает два лица по всем характеристикам с детальными результатами для каждого элемента
        Автоматически выравнивает лица перед сравнением для устойчивости к наклонам
        
        Args:
            features1: Характеристики первого лица
            features2: Характеристики второго лица
            
        Returns:
            Словарь с детальными результатами для каждой характеристики
        """
        # Автоматически выравниваем лица по углу наклона
        # Вычисляем углы наклона для обоих лиц
        angle1 = self.analyzer.calculate_face_angle(features1)
        angle2 = self.analyzer.calculate_face_angle(features2)
        
        # Выравниваем оба лица (поворачиваем к горизонтали)
        if abs(angle1) > 0.5:
            features1 = self.align_features_by_rotation(features1, -angle1)
        if abs(angle2) > 0.5:
            features2 = self.align_features_by_rotation(features2, -angle2)
        
        # Нормализуем размеры лиц перед сравнением
        features1, features2 = self.normalize_face_size(features1, features2)
        
        detailed_results = {}
        
        # Референсные точки для нормализации (овал лица)
        ref1 = features1.get('face_oval', np.array([]))
        ref2 = features2.get('face_oval', np.array([]))
        
        # Список характеристик для сравнения
        feature_names = [
            'face_oval', 'head_shape', 'left_eye', 'right_eye',
            'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip',
            'nose_contour', 'mouth_outer', 'mouth_inner', 'upper_lip',
            'lower_lip', 'left_cheek', 'right_cheek', 'left_ear',
            'right_ear', 'left_ear_detail', 'right_ear_detail', 'chin', 'forehead',
            'hair', 'hairline', 'left_temple', 'right_temple'  # Новые характеристики для волос
        ]
        
        for feature_name in feature_names:
            points1 = features1.get(feature_name, np.array([]))
            points2 = features2.get(feature_name, np.array([]))
            
            feature_result = {
                'similarity': 0.0,
                'point_similarity': 0.0,
                'metrics_similarity': 0.0,
                'has_data': False,
                'points_count_1': len(points1),
                'points_count_2': len(points2)
            }
            
            if len(points1) > 0 and len(points2) > 0:
                feature_result['has_data'] = True
                
                # Сравниваем точки
                point_similarity = self.compare_points(
                    points1, points2, 
                    normalize=True, 
                    reference1=ref1, 
                    reference2=ref2
                )
                feature_result['point_similarity'] = point_similarity
                
                # Вычисляем метрики
                metrics1 = self.analyzer.calculate_feature_metrics(points1)
                metrics2 = self.analyzer.calculate_feature_metrics(points2)
                
                # Сравниваем метрики
                metrics_similarity = self.compare_metrics(metrics1, metrics2)
                feature_result['metrics_similarity'] = metrics_similarity
                
                # Комбинируем результаты
                combined_similarity = (point_similarity * 0.65 + metrics_similarity * 0.35)
                
                # Небольшая коррекция для очень похожих лиц (обе метрики > 60%)
                # Это помогает распознавать одно и то же лицо при разных условиях съемки
                if point_similarity > 60 and metrics_similarity > 60:
                    combined_similarity = min(100, combined_similarity * 1.15)
                
                feature_result['similarity'] = min(100, combined_similarity)
                
                # Добавляем детали метрик
                feature_result['metrics_details'] = {
                    'area_ratio': min(metrics1.get('area', 0), metrics2.get('area', 0)) / 
                                 max(metrics1.get('area', 1), metrics2.get('area', 1)) if 
                                 metrics1.get('area', 0) > 0 and metrics2.get('area', 0) > 0 else 0,
                    'aspect_ratio_1': metrics1.get('aspect_ratio', 0),
                    'aspect_ratio_2': metrics2.get('aspect_ratio', 0)
                }
            
            detailed_results[feature_name] = feature_result
        
        # Вычисляем общий процент совпадения
        valid_similarities = [r['similarity'] for r in detailed_results.values() if r['has_data']]
        if valid_similarities:
            overall = np.mean(valid_similarities)
        else:
            overall = 0.0
        
        detailed_results['overall'] = {
            'similarity': overall,
            'features_compared': len(valid_similarities),
            'total_features': len(feature_names)
        }
        
        return detailed_results

