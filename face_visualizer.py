"""
Analytical System - Face Visualizer Module
Face comparison visualization

Repository: https://github.com/The-Open-Tech-Company/analytical-system
License: Unlicense (Open Source)
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple


class FaceVisualizer:
    """Класс для визуализации биометрических линий и результатов сравнения"""
    
    def __init__(self):
        # Цвета для визуализации
        self.color_green = (0, 255, 0)  # Зеленый для первого лица
        self.color_red = (0, 0, 255)    # Красный для второго лица
        self.color_yellow = (0, 255, 255)  # Желтый для совпадений
        self.color_white = (255, 255, 255)  # Белый фон
        
        # Толщина линий - увеличена для лучшей видимости
        self.line_thickness = 3  # Увеличено для лучшей видимости
        self.point_radius = 4  # Увеличено для лучшей видимости точек
    
    def draw_feature(self, image: np.ndarray, points: np.ndarray, 
                    color: Tuple[int, int, int], closed: bool = True) -> np.ndarray:
        """
        Рисует характеристику лица на изображении
        
        Args:
            image: Изображение для рисования
            points: Точки характеристики
            color: Цвет линий
            closed: Замкнут ли контур
            
        Returns:
            Изображение с нарисованными линиями
        """
        if len(points) < 2:
            return image
        
        image_copy = image.copy()
        
        # Проверяем, что points не пустой
        if points is None or len(points) < 2:
            return image_copy
        
        # Убеждаемся, что points - это numpy array
        if not isinstance(points, np.ndarray):
            try:
                # Пытаемся создать массив из точек
                points_list = []
                for p in points:
                    if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
                        try:
                            x = float(p[0])
                            y = float(p[1])
                            points_list.append([x, y])
                        except (ValueError, TypeError, IndexError):
                            continue
                
                if len(points_list) < 2:
                    return image_copy
                
                points = np.array(points_list, dtype=np.float64)
            except (ValueError, TypeError, IndexError):
                return image_copy
        
        # Проверяем, что points не пустой и имеет правильную форму
        if points.size == 0 or len(points) < 2:
            return image_copy
        
        # Проверяем, что points имеет правильную форму (N, 2)
        if len(points.shape) != 2 or points.shape[1] != 2:
            return image_copy
        
        # Конвертируем в int32 для рисования
        try:
            # Убеждаемся, что все значения числовые
            points_clean = np.array([[float(p[0]), float(p[1])] for p in points], dtype=np.float64)
            # Проверяем на NaN и Inf
            if np.any(np.isnan(points_clean)) or np.any(np.isinf(points_clean)):
                return image_copy
            points_int = points_clean.astype(np.int32)
        except (ValueError, TypeError, IndexError, OverflowError) as e:
            # Если не удалось конвертировать, возвращаем исходное изображение
            return image_copy
        
        # Рисуем линии (только линии, без точек для чистоты визуализации)
        if closed and len(points_int) >= 3:
            cv2.polylines(image_copy, [points_int], isClosed=True, 
                         color=color, thickness=self.line_thickness)
        else:
            for i in range(len(points_int) - 1):
                p1 = (int(points_int[i][0]), int(points_int[i][1]))
                p2 = (int(points_int[i + 1][0]), int(points_int[i + 1][1]))
                cv2.line(image_copy, p1, p2, color, self.line_thickness)
        
        # Не рисуем точки для упрощения визуализации (только линии)
        
        return image_copy
    
    def visualize_face_features(self, image: np.ndarray, features: Dict,
                               color: Tuple[int, int, int], max_points_per_feature: int = None) -> np.ndarray:
        """
        Визуализирует все характеристики лица на изображении
        
        Args:
            image: Исходное изображение
            features: Словарь с характеристиками лица
            color: Цвет для рисования
            max_points_per_feature: Максимальное количество точек на черту (None = без ограничения)
            
        Returns:
            Изображение с нарисованными характеристиками
        """
        result_image = image.copy()
        
        # Порядок рисования характеристик (от более крупных к мелким)
        feature_order = [
            'head_shape', 'face_oval', 'forehead', 'chin',
            'left_cheek', 'right_cheek',
            'left_ear', 'right_ear', 'left_ear_detail', 'right_ear_detail',
            'left_eyebrow', 'right_eyebrow',
            'left_eye', 'right_eye',
            'nose_bridge', 'nose_tip', 'nose_contour',
            'mouth_outer', 'mouth_inner', 'upper_lip', 'lower_lip'
        ]
        
        for feature_name in feature_order:
            points = features.get(feature_name, np.array([]))
            if len(points) > 0:
                # Применяем прореживание точек, если указано
                if max_points_per_feature is not None and max_points_per_feature > 0:
                    points = self._downsample_points(points, max_points_per_feature)
                
                closed = feature_name in ['face_oval', 'head_shape', 'left_eye', 
                                         'right_eye', 'mouth_outer', 'mouth_inner',
                                         'upper_lip', 'lower_lip', 'nose_contour']
                result_image = self.draw_feature(result_image, points, color, closed)
        
        return result_image
    
    def _downsample_points(self, points: np.ndarray, max_points: int = 100) -> np.ndarray:
        """
        Прореживает точки для визуализации (убирает избыточные точки после интерполяции)
        
        Args:
            points: Исходные точки
            max_points: Максимальное количество точек для визуализации
            
        Returns:
            Прореженные точки
        """
        if len(points) <= max_points:
            return points
        
        # Равномерно выбираем точки
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        return points[indices]
    
    def create_overlay_comparison(self, features1: Dict, features2: Dict,
                                 image1: np.ndarray, image2: np.ndarray,
                                 results: Dict[str, float]) -> np.ndarray:
        """
        Создает изображение с наложением двух рисунков линий на белом фоне
        Показывает все характеристики лица (зеленая и красная линии) с нормализацией размеров
        
        Args:
            features1: Характеристики первого лица
            features2: Характеристики второго лица
            image1: Первое изображение
            image2: Второе изображение
            results: Результаты сравнения
            
        Returns:
            Изображение с наложенными линиями
        """
        # Нормализуем размеры лиц перед визуализацией
        from face_comparator import FaceComparator
        comparator = FaceComparator()
        normalized_features1, normalized_features2 = comparator.normalize_face_size(features1, features2)
        
        # Определяем размер изображения
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        max_h = max(h1, h2)
        max_w = max(w1, w2)
        canvas_size = int(max(max_h, max_w) * 1.2)
        
        # Создаем белый фон
        overlay = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        
        # Получаем нормализованные овалы лиц для определения масштаба и центрирования
        face_oval1 = normalized_features1.get('face_oval', np.array([]))
        face_oval2 = normalized_features2.get('face_oval', np.array([]))
        
        if len(face_oval1) > 0 and len(face_oval2) > 0:
            # Прореживаем точки для визуализации (убираем избыточные после интерполяции)
            face_oval1 = self._downsample_points(face_oval1, max_points=100)
            face_oval2 = self._downsample_points(face_oval2, max_points=100)
            
            centroid1 = np.mean(face_oval1, axis=0)
            centroid2 = np.mean(face_oval2, axis=0)
            centroid1 = np.array([float(centroid1[0]), float(centroid1[1])])
            centroid2 = np.array([float(centroid2[0]), float(centroid2[1])])
            
            # Вычисляем размеры нормализованных овалов (они должны быть примерно одинаковыми)
            size1 = np.max(face_oval1, axis=0) - np.min(face_oval1, axis=0)
            size2 = np.max(face_oval2, axis=0) - np.min(face_oval2, axis=0)
            avg_size = float((float(np.mean(size1)) + float(np.mean(size2))) / 2.0)
            
            # Используем одинаковый масштаб для обоих (так как размеры уже нормализованы)
            scale = float((canvas_size * 0.75) / avg_size) if avg_size > 0 else 1.0
            
            # Центр канваса
            center_x = canvas_size // 2
            center_y = canvas_size // 2
            
            # Смещения для центрирования
            offset_x1 = float(center_x - float(centroid1[0]) * scale)
            offset_y1 = float(center_y - float(centroid1[1]) * scale)
            offset_x2 = float(center_x - float(centroid2[0]) * scale)
            offset_y2 = float(center_y - float(centroid2[1]) * scale)
            
            # Порядок рисования характеристик (от более крупных к мелким)
            feature_order = [
                'head_shape', 'face_oval', 'forehead', 'chin',
                'left_cheek', 'right_cheek',
                'left_ear', 'right_ear', 'left_ear_detail', 'right_ear_detail',
                'left_eyebrow', 'right_eyebrow',
                'left_eye', 'right_eye',
                'nose_bridge', 'nose_tip', 'nose_contour',
                'mouth_outer', 'mouth_inner', 'upper_lip', 'lower_lip'
            ]
            
            # Определяем, какие характеристики замкнуты
            closed_features = ['face_oval', 'head_shape', 'left_eye', 'right_eye', 
                             'mouth_outer', 'mouth_inner', 'upper_lip', 'lower_lip', 
                             'nose_contour', 'left_ear', 'right_ear']
            
            # Рисуем все характеристики первого лица (зеленый) - рисуем с небольшим смещением влево
            for feature_name in feature_order:
                points1 = normalized_features1.get(feature_name, np.array([]))
                if len(points1) > 0:
                    # Прореживаем точки для визуализации
                    points1 = self._downsample_points(points1, max_points=100)
                    
                    # Масштабируем и смещаем точки (с небольшим смещением влево для видимости)
                    scaled_points1 = points1.astype(np.float64) * float(scale)
                    scaled_points1[:, 0] += float(offset_x1) - 5  # Смещение влево на 5 пикселей
                    scaled_points1[:, 1] += float(offset_y1)
                    
                    # Определяем, замкнут ли контур
                    closed = feature_name in closed_features
                    
                    # Рисуем характеристику с увеличенной толщиной для лучшей видимости
                    overlay = self.draw_feature(overlay, scaled_points1, self.color_green, closed)
            
            # Рисуем все характеристики второго лица (красный) - рисуем с небольшим смещением вправо
            for feature_name in feature_order:
                points2 = normalized_features2.get(feature_name, np.array([]))
                if len(points2) > 0:
                    # Прореживаем точки для визуализации
                    points2 = self._downsample_points(points2, max_points=100)
                    
                    # Масштабируем и смещаем точки (с небольшим смещением вправо для видимости)
                    scaled_points2 = points2.astype(np.float64) * float(scale)
                    scaled_points2[:, 0] += float(offset_x2) + 5  # Смещение вправо на 5 пикселей
                    scaled_points2[:, 1] += float(offset_y2)
                    
                    # Определяем, замкнут ли контур
                    closed = feature_name in closed_features
                    
                    # Рисуем характеристику с увеличенной толщиной для лучшей видимости
                    overlay = self.draw_feature(overlay, scaled_points2, self.color_red, closed)
        else:
            # Fallback: если нет овалов, не рисуем ничего
            pass
        
        return overlay
    
    def _draw_matches(self, overlay: np.ndarray, features1: Dict, features2: Dict,
                     scale1: float, scale2: float,
                     offset_x1: int, offset_y1: int, offset_x2: int, offset_y2: int,
                     results: Dict[str, float]):
        """Рисует области совпадения желтым цветом"""
        threshold = 70  # Порог для определения совпадения (%)
        
        for feature_name, similarity in results.items():
            if feature_name == 'overall':
                continue
            
            if similarity >= threshold:
                points1 = features1.get(feature_name, np.array([]))
                points2 = features2.get(feature_name, np.array([]))
                
                # Проверяем, что points1 и points2 - это numpy arrays
                if (isinstance(points1, np.ndarray) and isinstance(points2, np.ndarray) and 
                    len(points1) > 0 and len(points2) > 0):
                    try:
                        # Масштабируем точки первого лица
                        scaled_points1 = points1.astype(np.float64) * float(scale1)
                        scaled_points1[:, 0] += float(offset_x1)
                        scaled_points1[:, 1] += float(offset_y1)
                        
                        # Масштабируем точки второго лица
                        scaled_points2 = points2.astype(np.float64) * float(scale2)
                        scaled_points2[:, 0] += float(offset_x2)
                        scaled_points2[:, 1] += float(offset_y2)
                        
                        # Рисуем среднюю линию между двумя характеристиками
                        if len(scaled_points1) == len(scaled_points2):
                            for i in range(len(scaled_points1)):
                                try:
                                    p1 = scaled_points1[i].astype(np.int32)
                                    p2 = scaled_points2[i].astype(np.int32)
                                    mid_point = ((p1 + p2) // 2).astype(np.int32)
                                    pt = (int(mid_point[0]), int(mid_point[1]))
                                    cv2.circle(overlay, pt, 2, self.color_yellow, -1)
                                except (ValueError, TypeError, OverflowError):
                                    continue
                    except (ValueError, TypeError, AttributeError, OverflowError):
                        continue
    
    def create_results_image(self, results: Dict[str, float], 
                           width: int = 800, height: int = 600) -> np.ndarray:
        """
        Создает изображение с текстовыми результатами сравнения
        
        Args:
            results: Словарь с результатами сравнения
            width: Ширина изображения
            height: Высота изображения
            
        Returns:
            Изображение с результатами
        """
        # Создаем белое изображение
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Русские названия характеристик
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
            'overall': 'ОБЩЕЕ СОВПАДЕНИЕ'
        }
        
        # Параметры текста
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 30
        start_y = 40
        start_x = 20
        
        y = start_y
        
        # Заголовок
        cv2.putText(img, 'РЕЗУЛЬТАТЫ СРАВНЕНИЯ ЛИЦ', 
                   (start_x, y), font, 1.0, (0, 0, 0), 2)
        y += line_height * 2
        
        # Выводим результаты для каждой характеристики
        for feature_name, similarity in results.items():
            if feature_name == 'overall':
                continue
            
            name_ru = feature_names_ru.get(feature_name, feature_name)
            text = f'{name_ru}: {similarity:.1f}%'
            
            # Цвет в зависимости от процента совпадения
            if similarity >= 80:
                color = (0, 200, 0)  # Зеленый
            elif similarity >= 60:
                color = (0, 165, 255)  # Оранжевый
            else:
                color = (0, 0, 200)  # Красный
            
            cv2.putText(img, text, (start_x, y), font, font_scale, color, thickness)
            y += line_height
        
        # Общий результат (выделяем)
        y += line_height
        overall = results.get('overall', 0.0)
        text = f'{feature_names_ru.get("overall", "Общее совпадение")}: {overall:.1f}%'
        
        # Цвет для общего результата
        if overall >= 80:
            color = (0, 200, 0)
        elif overall >= 60:
            color = (0, 165, 255)
        else:
            color = (0, 0, 200)
        
        cv2.putText(img, text, (start_x, y), font, 1.2, color, 3)
        
        return img

