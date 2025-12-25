"""
Analytical System - Face Analyzer Module
Face feature extraction and analysis using MediaPipe

Repository: https://github.com/The-Open-Tech-Company/analytical-system
License: Unlicense (Open Source)
"""
import os
import sys
import logging
import warnings
from contextlib import redirect_stderr
from io import StringIO

# Подавление предупреждений MediaPipe/TensorFlow Lite (должно быть ДО импорта mediapipe)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Подавляет все сообщения от TensorFlow (0=все, 1=INFO, 2=WARNING, 3=ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключает оптимизации, которые могут вызывать предупреждения

# Подавляем все предупреждения Python
warnings.filterwarnings('ignore')

# Настраиваем логирование
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Подавляем stderr при импорте mediapipe
_stderr_buffer = StringIO()
with redirect_stderr(_stderr_buffer):
    import cv2
    import numpy as np
    import mediapipe as mp
from typing import Dict, List, Tuple, Optional

# Импортируем модуль для работы с OpenCV DNN
try:
    from gender_age_dnn import GenderAgeDNN
    DNN_AVAILABLE = True
except ImportError:
    DNN_AVAILABLE = False
    if logger:
        logger.warning("Модуль gender_age_dnn не найден. Будет использован эвристический метод.")

# Импортируем модуль для работы с DeepFace (приоритетный метод)
try:
    from deepface_gender_analyzer import DeepFaceGenderAnalyzer
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    if logger:
        logger.warning("Модуль deepface_gender_analyzer не найден. DeepFace будет недоступен.")

# Импортируем продвинутый детектор пола (максимальная точность)
try:
    from advanced_gender_detector import AdvancedGenderDetector
    ADVANCED_GENDER_AVAILABLE = True
except ImportError:
    ADVANCED_GENDER_AVAILABLE = False
    if logger:
        logger.warning("Модуль advanced_gender_detector не найден. Будет использован стандартный метод.")

# Импортируем модуль для определения расы (приоритетный метод)
try:
    from race_analyzer import RaceAnalyzer
    RACE_ANALYZER_AVAILABLE = True
except ImportError:
    RACE_ANALYZER_AVAILABLE = False
    if logger:
        logger.warning("Модуль race_analyzer не найден. Будет использован эвристический метод для расы.")


class FaceAnalyzer:
    """Класс для анализа лица с использованием MediaPipe Face Mesh"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        # Снижаем порог детекции для работы с плохими фото
        # Подавляем stderr при инициализации FaceMesh
        _stderr_buffer = StringIO()
        with redirect_stderr(_stderr_buffer):
            # Face Detection для предварительного обнаружения лиц
            # Обновлено для MediaPipe 0.10.21 - улучшенная совместимость
            try:
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=1,  # 0 для ближних лиц, 1 для дальних
                    min_detection_confidence=0.2  # Очень низкий порог
                )
            except Exception as e:
                logger.warning(f"Ошибка при инициализации FaceDetection: {e}. Используем значения по умолчанию.")
                self.face_detection = self.mp_face_detection.FaceDetection(
                    min_detection_confidence=0.2
                )
            
            # Face Mesh для детального анализа
            # Обновлено для MediaPipe 0.10.21 - поддержка refine_landmarks
            try:
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,  # Улучшенные landmarks в новых версиях
                    min_detection_confidence=0.2,  # Еще более сниженный порог
                    min_tracking_confidence=0.2
                )
            except TypeError:
                # Если refine_landmarks не поддерживается в старых версиях
                logger.debug("refine_landmarks не поддерживается, используем без него")
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    min_detection_confidence=0.2,
                    min_tracking_confidence=0.2
                )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Индексы ключевых точек для различных характеристик
        self.landmark_indices = self._get_landmark_indices()
        
        # Инициализируем продвинутый детектор пола (наивысший приоритет - использует ансамбль методов)
        self.advanced_gender_detector = None
        if ADVANCED_GENDER_AVAILABLE:
            try:
                self.advanced_gender_detector = AdvancedGenderDetector()
                if not self.advanced_gender_detector.is_available():
                    logger.warning("Продвинутый детектор пола недоступен. Будет использован DeepFace.")
                    self.advanced_gender_detector = None
                else:
                    logger.info("Продвинутый детектор пола успешно инициализирован (ансамбль методов)")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать продвинутый детектор пола: {e}")
                self.advanced_gender_detector = None
        
        # Инициализируем модуль для определения пола и возраста через DeepFace (резервный метод)
        self.deepface_analyzer = None
        if DEEPFACE_AVAILABLE:
            try:
                # Используем ArcFace как модель по умолчанию для максимальной точности
                # В DeepFace 0.0.95 ArcFace показывает лучшие результаты для определения пола и возраста
                # Можно также использовать "VGG-Face" для баланса точности и скорости
                try:
                    self.deepface_analyzer = DeepFaceGenderAnalyzer(model_name="ArcFace")
                    if not self.deepface_analyzer.is_available():
                        # Пробуем VGG-Face как резервный вариант
                        logger.warning("ArcFace недоступен, пробуем VGG-Face...")
                        self.deepface_analyzer = DeepFaceGenderAnalyzer(model_name="VGG-Face")
                except Exception:
                    # Если ArcFace не поддерживается, используем VGG-Face
                    self.deepface_analyzer = DeepFaceGenderAnalyzer(model_name="VGG-Face")
                
                if not self.deepface_analyzer.is_available():
                    logger.warning("DeepFace недоступен. Будет использован OpenCV DNN или эвристический метод.")
                else:
                    logger.info("DeepFace успешно инициализирован для определения пола и возраста")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать DeepFace: {e}. Будет использован OpenCV DNN или эвристический метод.")
                self.deepface_analyzer = None
        
        # Инициализируем модуль для определения пола и возраста через OpenCV DNN (резервный метод)
        self.dnn_analyzer = None
        if DNN_AVAILABLE:
            try:
                # Пробуем стандартный порядок классов [Male, Female] сначала
                # Если результаты неправильные, можно попробовать swap_gender_classes=True
                self.dnn_analyzer = GenderAgeDNN(swap_gender_classes=False)
                if not self.dnn_analyzer.is_available():
                    logger.warning("Модели OpenCV DNN не загружены. Будет использован эвристический метод.")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать OpenCV DNN: {e}. Будет использован эвристический метод.")
                self.dnn_analyzer = None
        
        # Инициализируем модуль для определения расы (приоритетный метод - DeepFace/InsightFace)
        self.race_analyzer = None
        if RACE_ANALYZER_AVAILABLE:
            try:
                self.race_analyzer = RaceAnalyzer()
                if not self.race_analyzer.is_available():
                    logger.warning("Анализатор расы недоступен. Будет использован эвристический метод.")
                else:
                    logger.info("Анализатор расы успешно инициализирован (DeepFace/InsightFace)")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать анализатор расы: {e}. Будет использован эвристический метод.")
                self.race_analyzer = None
    
    def _get_landmark_indices(self) -> Dict[str, List[int]]:
        """Определяет индексы ключевых точек для различных характеристик лица"""
        return {
            # Овал лица (контур) - значительно расширено для более детального сравнения
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 151, 337, 299, 333, 298, 301,
                         368, 264, 447, 366, 401, 435, 410, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                         152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            
            # Левый глаз - значительно расширено для более детального сравнения
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 
                        247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25, 130, 243, 112,
                        26, 22, 23, 24, 110, 25, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161,
                        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 247, 30, 29, 27, 28, 56, 190, 243, 112],
            'left_eyebrow': [46, 53, 52, 65, 55, 70, 63, 105, 66, 107, 31, 228, 229, 230, 231, 232, 233,
                           46, 53, 52, 65, 55, 70, 63, 105, 66, 107, 31, 228, 229, 230, 231, 232, 233],
            
            # Правый глаз - значительно расширено для более детального сравнения
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
                         359, 255, 339, 254, 253, 252, 256, 341, 463, 414, 286, 258, 257, 259, 260, 467,
                         260, 259, 257, 258, 286, 414, 463, 341, 256, 252, 253, 254, 339, 255, 359, 398,
                         384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362],
            'right_eyebrow': [276, 283, 282, 295, 285, 300, 293, 334, 296, 336, 276, 283, 282, 295, 285, 300, 293,
                             276, 283, 282, 295, 285, 300, 293, 334, 296, 336, 276, 283, 282, 295, 285, 300, 293],
            
            # Нос - значительно расширено для более детального сравнения
            'nose_bridge': [6, 51, 48, 115, 131, 134, 102, 49, 220, 305, 290, 305, 4, 5, 6, 19, 20, 94, 125, 141,
                           6, 51, 48, 115, 131, 134, 102, 49, 220, 305, 290, 305, 4, 5, 6, 19, 20, 94, 125, 141],
            'nose_tip': [1, 2, 5, 4, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 6, 102, 49, 220, 305, 290,
                        1, 2, 5, 4, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 6, 102, 49, 220, 305, 290],
            'nose_contour': [49, 131, 134, 102, 49, 220, 305, 290, 305, 4, 5, 6, 19, 20, 1, 2, 3, 51, 48, 115, 125, 141, 235, 236,
                            49, 131, 134, 102, 49, 220, 305, 290, 305, 4, 5, 6, 19, 20, 1, 2, 3, 51, 48, 115, 125, 141, 235, 236],
            
            # Рот - значительно расширено для более детального сравнения
            'mouth_outer': [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 
                          12, 15, 16, 17, 18, 200, 269, 270, 271, 272, 407, 408, 409, 415, 310, 311, 312, 13,
                          61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 12, 15, 16, 17, 18],
            'mouth_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 12, 15, 16, 17, 18, 200, 269, 270, 271, 272,
                           78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 12, 15, 16, 17, 18, 200, 269, 270, 271, 272],
            'upper_lip': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 12, 15, 16, 13, 82, 81, 80, 78,
                         61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 12, 15, 16, 13, 82, 81, 80, 78],
            'lower_lip': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 12, 15, 16, 17, 18, 200, 269, 270, 271, 272,
                         78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 12, 15, 16, 17, 18, 200, 269, 270, 271, 272],
            
            # Скулы - значительно расширено для более детального сравнения
            'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 50, 101, 100, 99, 98, 97, 2, 326, 327, 328, 329, 330, 331,
                          116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 50, 101, 100, 99, 98, 97, 2, 326, 327, 328, 329, 330, 331],
            'right_cheek': [345, 346, 347, 348, 349, 350, 351, 352, 266, 425, 426, 427, 280, 344, 340, 352, 376, 411, 427, 422, 432, 436, 427,
                           345, 346, 347, 348, 349, 350, 351, 352, 266, 425, 426, 427, 280, 344, 340, 352, 376, 411, 427, 422, 432, 436, 427],
            
            # Уши (силуэт и контур) - используем точки контура лица в области ушей
            # Левое ухо - точки левой стороны лица в области уха (висок, за ухом, нижняя часть)
            # Индексы из контура лица, которые находятся в области левого уха
            'left_ear': [234, 127, 162, 21, 54, 103, 67, 109, 10, 151, 9, 10, 338, 297, 332, 284],
            # Правое ухо - точки правой стороны лица в области уха (висок, за ухом, нижняя часть)
            # Индексы из контура лица, которые находятся в области правого уха
            'right_ear': [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136],
            # Дополнительные точки для левого уха (верхняя часть уха, область виска)
            'left_ear_detail': [234, 127, 162, 21, 54, 103, 67, 109],
            # Дополнительные точки для правого уха (верхняя часть уха, область виска)
            'right_ear_detail': [454, 323, 361, 288, 397, 365, 379, 378],
            
            # Подбородок - значительно расширено для более детального сравнения
            'chin': [18, 200, 199, 175, 169, 170, 140, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323,
                    18, 200, 199, 175, 169, 170, 140, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
            
            # Лоб - значительно расширено для более детального сравнения
            'forehead': [10, 151, 9, 10, 151, 337, 299, 333, 298, 301, 338, 297, 332, 284, 251, 389, 356, 454,
                        10, 151, 9, 10, 151, 337, 299, 333, 298, 301, 338, 297, 332, 284, 251, 389, 356, 454,
                        368, 264, 447, 366, 401, 435, 410, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377],
            
            # Волосы - новая характеристика для сравнения прически и линии роста волос
            # Используем точки верхней части головы, лба, висков и контура головы в области волос
            'hair': [10, 151, 9, 10, 151, 337, 299, 333, 298, 301, 338, 297, 332, 284, 251, 389, 356, 454,
                    368, 264, 447, 366, 401, 435, 410, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
                    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            
            # Линия роста волос (верхняя граница) - для сравнения формы линии роста волос
            'hairline': [10, 151, 9, 10, 151, 337, 299, 333, 298, 301, 338, 297, 332, 284, 251, 389, 356, 454,
                        368, 264, 447, 366, 401, 435, 410, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            
            # Висок левый - для сравнения формы виска и прически
            'left_temple': [234, 127, 162, 21, 54, 103, 67, 109, 10, 151, 9, 10, 338, 297, 332, 284,
                          234, 127, 162, 21, 54, 103, 67, 109, 10, 151, 9, 10, 338, 297, 332, 284,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            
            # Висок правый - для сравнения формы виска и прически
            'right_temple': [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                           454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                           172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            
            # Форма головы (общий контур) - значительно расширено для более детального сравнения
            'head_shape': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
                          151, 337, 299, 333, 298, 301, 9, 10, 151, 9, 10, 338, 297, 332, 284,
                          368, 264, 447, 366, 401, 435, 410, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                          152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Предобрабатывает изображение для улучшения качества детекции на плохих фото
        Автоматически определяет плохое качество и увеличивает контрастность/яркость
        
        Args:
            image: Исходное изображение
            
        Returns:
            Обработанное изображение
        """
        # Конвертируем в RGB если нужно
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Определяем качество изображения (яркость и контрастность)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            std_contrast = np.std(gray)
            
            # Определяем, нужно ли улучшение (темное или низкоконтрастное изображение)
            needs_brightness_boost = mean_brightness < 100  # Темное изображение
            needs_contrast_boost = std_contrast < 40  # Низкоконтрастное изображение
            
            # Если изображение темное или низкоконтрастное, применяем усиленную обработку
            if needs_brightness_boost or needs_contrast_boost:
                # Усиленное улучшение контраста с помощью CLAHE
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Более агрессивное CLAHE для плохих фото
                clahe_clip_limit = 3.0 if needs_contrast_boost else 2.0
                clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # Увеличиваем яркость, если изображение темное
                if needs_brightness_boost:
                    brightness_boost = int((100 - mean_brightness) * 0.5)  # Адаптивное увеличение яркости
                    l = cv2.add(l, brightness_boost)
                
                # Объединяем каналы обратно
                lab = cv2.merge([l, a, b])
                image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
                # Дополнительное увеличение контрастности через линейное преобразование
                if needs_contrast_boost:
                    alpha = 1.5  # Коэффициент контрастности
                    beta = 10   # Смещение яркости
                    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            else:
                # Стандартная обработка для нормальных изображений
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Применяем CLAHE к каналу яркости
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # Объединяем каналы обратно
                lab = cv2.merge([l, a, b])
                image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Легкое шумоподавление (сохраняем детали)
            image = cv2.bilateralFilter(image, 5, 50, 50)
            
            # Усиленное увеличение резкости для плохих фото
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            sharpening_strength = 0.15 if (needs_brightness_boost or needs_contrast_boost) else 0.1
            image = cv2.filter2D(image, -1, kernel * sharpening_strength) + image * (1 - sharpening_strength)
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def extract_face_features(self, image_path: str) -> Optional[Dict]:
        """
        Извлекает характеристики лица из изображения
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Словарь с характеристиками лица или None, если лицо не найдено
        """
        # Пробуем разные способы загрузки изображения
        image = cv2.imread(image_path)
        if image is None:
            # Пробуем через numpy (для путей с кириллицей)
            try:
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except:
                pass
        
        if image is None:
            # Пробуем через PIL
            try:
                from PIL import Image
                pil_img = Image.open(image_path)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img_array = np.array(pil_img)
                image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            except:
                pass
        
        if image is None:
            return None
        
        # Сохраняем оригинальное изображение
        original_image = image.copy()
        h, w = original_image.shape[:2]
        
        # Пробуем разные варианты детекции
        results = None
        image_to_use = None
        
        # Список вариантов для попытки детекции
        detection_variants = [
            # 1. Оригинальное изображение
            (original_image, "original"),
            # 2. Предобработанное изображение
            (self.preprocess_image(original_image.copy()), "preprocessed"),
            # 3. Увеличенное изображение (если маленькое)
            (cv2.resize(original_image, (w*2, h*2), interpolation=cv2.INTER_LINEAR) if w < 500 or h < 500 else None, "upscaled"),
            # 4. Уменьшенное изображение (если большое)
            (cv2.resize(original_image, (w//2, h//2), interpolation=cv2.INTER_LINEAR) if w > 2000 or h > 2000 else None, "downscaled"),
            # 5. Изображение с улучшенной яркостью
            (cv2.convertScaleAbs(original_image, alpha=1.2, beta=10), "brightened"),
            # 6. Изображение с улучшенным контрастом
            (cv2.convertScaleAbs(original_image, alpha=1.3, beta=0), "contrasted"),
        ]
        
        # Пробуем каждый вариант
        for variant_image, variant_name in detection_variants:
            if variant_image is None:
                continue
            
            # Конвертируем в RGB
            image_rgb = cv2.cvtColor(variant_image, cv2.COLOR_BGR2RGB)
            
            # Сначала пробуем Face Detection для проверки наличия лица
            detection_results = self.face_detection.process(image_rgb)
            if detection_results.detections:
                # Если Face Detection нашел лицо, пробуем Face Mesh
                results = self.face_mesh.process(image_rgb)
                if results.multi_face_landmarks:
                    image_to_use = variant_image
                    break
            else:
                # Пробуем Face Mesh напрямую (иногда он находит то, что не находит Detection)
                results = self.face_mesh.process(image_rgb)
                if results.multi_face_landmarks:
                    image_to_use = variant_image
                    break
        
        # Если ничего не помогло, пробуем еще раз с оригиналом и очень низким порогом
        if not results or not results.multi_face_landmarks:
            image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            image_to_use = original_image
        
        if not results or not results.multi_face_landmarks:
            return None
        
        image = image_to_use if image_to_use is not None else original_image
        
        face_landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Проверяем качество детекции - если слишком мало точек, возвращаем None
        if len(face_landmarks.landmark) < 100:
            return None
        
        # Преобразуем landmarks в координаты пикселей
        landmarks_2d = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_2d.append((x, y))
        
        features = {}
        
        # Извлекаем характеристики
        for feature_name, indices in self.landmark_indices.items():
            # Фильтруем валидные индексы (MediaPipe имеет 468 точек)
            valid_indices = [i for i in indices if 0 <= i < len(landmarks_2d)]
            points = [landmarks_2d[i] for i in valid_indices]
            if points:
                points_array = np.array(points)
                # Увеличиваем количество точек в 30 раз через интерполяцию (для более детального сравнения)
                points_array = self._interpolate_points_10x(points_array)
                features[feature_name] = points_array
        
        # Сохраняем исходное изображение и landmarks для визуализации
        features['image'] = image
        features['landmarks_2d'] = landmarks_2d
        features['all_landmarks'] = face_landmarks
        
        # Определяем пол, возраст и расу
        gender_age_race = self.estimate_gender_and_age(features, image)
        features['gender'] = gender_age_race['gender']
        features['age'] = gender_age_race['age']
        features['race'] = gender_age_race['race']
        features['gender_confidence'] = gender_age_race['gender_confidence']
        features['age_confidence'] = gender_age_race['age_confidence']
        features['race_confidence'] = gender_age_race['race_confidence']
        features['confidence'] = gender_age_race['confidence']
        
        return features
    
    def get_feature_points(self, features: Dict, feature_name: str) -> np.ndarray:
        """Получает точки для конкретной характеристики"""
        return features.get(feature_name, np.array([]))
    
    def calculate_face_angle(self, features: Dict) -> float:
        """
        Вычисляет угол наклона головы на основе положения глаз
        Выравнивает так, чтобы глаза были на горизонтали
        
        Args:
            features: Словарь с характеристиками лица
            
        Returns:
            Угол наклона в градусах для выравнивания (положительный = поворот против часовой)
        """
        left_eye = features.get('left_eye', np.array([]))
        right_eye = features.get('right_eye', np.array([]))
        
        if len(left_eye) == 0 or len(right_eye) == 0:
            return 0.0
        
        # Вычисляем центры глаз
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        
        # Вычисляем вектор между глазами
        # В MediaPipe: left_eye - это левый глаз человека (справа на изображении для зрителя)
        # right_eye - это правый глаз человека (слева на изображении для зрителя)
        # Для выравнивания: если левый глаз выше правого (dy < 0), нужно повернуть по часовой
        # Если правый глаз выше левого (dy > 0), нужно повернуть против часовой
        
        # Определяем, какой глаз выше
        # left_eye_center[1] - это Y координата левого глаза (меньше = выше)
        # right_eye_center[1] - это Y координата правого глаза
        dy = float(left_eye_center[1] - right_eye_center[1])  # Разница по вертикали
        dx = float(right_eye_center[0] - left_eye_center[0])  # Разница по горизонтали
        
        # Если dx очень мал, значит лицо повернуто почти в профиль
        if abs(dx) < 10:
            return 0.0
        
        # Вычисляем угол наклона линии между глазами
        # Нам нужно повернуть так, чтобы dy стал близок к 0 (горизонталь)
        # atan2(dy, dx) даст угол наклона линии
        angle_rad = np.arctan2(float(dy), float(dx))
        angle_deg = float(np.degrees(angle_rad))
        
        # Угол для выравнивания - нужно повернуть на противоположный угол
        # Если линия наклонена вверх вправо (dy < 0, угол отрицательный), 
        # нужно повернуть по часовой (отрицательный угол поворота)
        return -angle_deg
    
    def align_face(self, image: np.ndarray, features: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Выравнивает лицо, поворачивая изображение так, чтобы глаза были на одной горизонтали
        
        Args:
            image: Исходное изображение
            features: Характеристики лица
            
        Returns:
            Кортеж (выровненное изображение, обновленные характеристики)
        """
        angle = self.calculate_face_angle(features)
        
        # Если угол очень мал, не поворачиваем
        if abs(angle) < 0.5:
            return image, features
        
        # Ограничиваем угол поворота разумными пределами (не более 45 градусов)
        if abs(angle) > 45:
            angle = 45 if angle > 0 else -45
        
        # Получаем размеры изображения
        h, w = image.shape[:2]
        
        # Используем центр лица (между глазами) как точку поворота
        left_eye = features.get('left_eye', np.array([]))
        right_eye = features.get('right_eye', np.array([]))
        
        if len(left_eye) > 0 and len(right_eye) > 0:
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            face_center = (float(left_eye_center[0] + right_eye_center[0]) / 2.0,
                          float(left_eye_center[1] + right_eye_center[1]) / 2.0)
        else:
            face_center = (float(w // 2), float(h // 2))
        
        # Создаем матрицу поворота вокруг центра лица
        rotation_matrix = cv2.getRotationMatrix2D(face_center, float(angle), 1.0)
        
        # Вычисляем новые размеры изображения с учетом поворота
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Корректируем матрицу поворота для учета новых размеров
        rotation_matrix[0, 2] += (new_w / 2) - face_center[0]
        rotation_matrix[1, 2] += (new_h / 2) - face_center[1]
        
        # Поворачиваем изображение с заполнением белым цветом вместо черного
        aligned_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(255, 255, 255))
        
        # Пересчитываем landmarks для выровненного изображения
        # Используем MediaPipe для повторного анализа выровненного изображения
        aligned_image_rgb = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(aligned_image_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h_new, w_new = aligned_image.shape[:2]
            
            # Преобразуем landmarks в координаты пикселей
            landmarks_2d = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w_new)
                y = int(landmark.y * h_new)
                landmarks_2d.append((x, y))
            
            # Извлекаем характеристики из выровненного изображения
            aligned_features = {}
            for feature_name, indices in self.landmark_indices.items():
                valid_indices = [i for i in indices if 0 <= i < len(landmarks_2d)]
                points = [landmarks_2d[i] for i in valid_indices]
                if points:
                    points_array = np.array(points)
                    # Увеличиваем количество точек в 30 раз через интерполяцию (для более детального сравнения)
                    points_array = self._interpolate_points_10x(points_array)
                    aligned_features[feature_name] = points_array
            
            aligned_features['landmarks_2d'] = landmarks_2d
            aligned_features['all_landmarks'] = face_landmarks
            aligned_features['image'] = aligned_image
        else:
            # Если MediaPipe не смог найти лицо после поворота, используем повернутые точки
            aligned_features = {}
            for feature_name, points in features.items():
                if feature_name in ['image', 'landmarks_2d', 'all_landmarks']:
                    continue
                
                if len(points) > 0:
                    # Преобразуем точки в однородные координаты
                    points_homogeneous = np.ones((len(points), 3))
                    points_homogeneous[:, :2] = points
                    
                # Применяем матрицу поворота
                aligned_points = (rotation_matrix @ points_homogeneous.T).T
                aligned_features[feature_name] = aligned_points[:, :2].astype(np.float64)
            
            aligned_features['image'] = aligned_image
        
        return aligned_image, aligned_features
    
    def _interpolate_points_10x(self, points: np.ndarray) -> np.ndarray:
        """
        Увеличивает количество точек в 30 раз через интерполяцию (ужесточено для более точного сравнения)
        
        Args:
            points: Исходные точки
            
        Returns:
            Массив точек с увеличенным количеством (в 30 раз больше)
        """
        if len(points) < 2:
            return points
        
        # Целевое количество точек (в 30 раз больше для более детального сравнения)
        target_count = len(points) * 30
        
        # Если точек уже достаточно, возвращаем как есть
        if len(points) >= target_count:
            return points
        
        # Определяем, замкнут ли контур (для овалов и других замкнутых форм)
        is_closed = len(points) >= 3
        
        # Создаем новые точки через интерполяцию
        new_points = []
        
        # Количество сегментов (для замкнутого контура = количество точек, для незамкнутого = количество точек - 1)
        num_segments = len(points) if is_closed else len(points) - 1
        
        if num_segments == 0:
            return points
        
        # Количество точек на сегмент (целевое количество / количество сегментов)
        points_per_segment = target_count // num_segments
        remainder = target_count % num_segments
        
        for i in range(len(points)):
            # Определяем следующую точку
            if i < len(points) - 1:
                p1 = points[i]
                p2 = points[i + 1]
            elif is_closed:
                # Для замкнутого контура последняя точка соединяется с первой
                p1 = points[-1]
                p2 = points[0]
            else:
                # Для незамкнутого контура просто добавляем последнюю точку
                new_points.append(points[-1])
                break
            
            # Добавляем начальную точку сегмента
            new_points.append(p1)
            
            # Количество промежуточных точек для этого сегмента
            segment_points = points_per_segment
            if i < remainder:  # Распределяем остаток по первым сегментам
                segment_points += 1
            
            # Добавляем промежуточные точки
            for j in range(1, segment_points):
                t = j / float(segment_points)  # Параметр интерполяции от 0 до 1
                interpolated = p1 + (p2 - p1) * t
                new_points.append(interpolated)
        
        # Если получилось больше точек, чем нужно, прореживаем равномерно
        if len(new_points) > target_count:
            indices = np.linspace(0, len(new_points) - 1, target_count, dtype=np.int32)
            new_points = [new_points[i] for i in indices]
        # Если получилось меньше, добавляем последнюю точку
        elif len(new_points) < target_count and len(points) > 0:
            while len(new_points) < target_count:
                new_points.append(points[-1])
        
        return np.array(new_points)
    
    def calculate_feature_metrics(self, points: np.ndarray) -> Dict:
        """
        Вычисляет метрики для набора точек характеристики
        
        Returns:
            Словарь с метриками: площадь, периметр, центроид, углы, расстояния
        """
        if len(points) < 3:
            return {}
        
        # Центроид
        centroid = np.mean(points, axis=0)
        centroid = np.array([float(centroid[0]), float(centroid[1])])
        
        # Площадь (для замкнутых контуров)
        if len(points) >= 3:
            try:
                # Убеждаемся, что points можно конвертировать в float32
                points_float32 = points.astype(np.float32)
                area = float(cv2.contourArea(points_float32))
            except (ValueError, TypeError, OverflowError):
                area = 0.0
        else:
            area = 0.0
        
        # Периметр
        try:
            points_float32 = points.astype(np.float32)
            perimeter = float(cv2.arcLength(points_float32, closed=True))
        except (ValueError, TypeError, OverflowError):
            perimeter = 0.0
        
        # Расстояния от центроида до каждой точки
        distances = [float(np.linalg.norm(p - centroid)) for p in points]
        
        # Углы между соседними точками
        angles = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            p3 = points[(i + 2) % len(points)]
            v1 = p2 - p1
            v2 = p3 - p2
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = float(np.arccos(cos_angle))
                angles.append(angle)
        
        # Соотношение сторон (bounding box)
        width = 0.0
        height = 0.0
        if len(points) > 0:
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            width = float(np.max(x_coords) - np.min(x_coords))
            height = float(np.max(y_coords) - np.min(y_coords))
            aspect_ratio = float(width / height) if height > 0 else 0.0
        else:
            aspect_ratio = 0.0
        
        return {
            'centroid': centroid,
            'area': area,
            'perimeter': perimeter,
            'distances': distances,
            'angles': angles,
            'aspect_ratio': aspect_ratio,
            'width': width,
            'height': height
        }
    
    def estimate_gender_and_age(self, features: Dict, image: np.ndarray) -> Dict[str, any]:
        """
        Оценивает пол, возраст и расу на основе DeepFace (приоритетный), OpenCV DNN или пропорций лица
        
        Args:
            features: Словарь с характеристиками лица
            image: Изображение лица
            
        Returns:
            Словарь с оценками пола, возраста и расы
        """
        result = {
            'gender': 'Не определен', 
            'age': 'Не определен', 
            'race': 'Не определена',
            'confidence': 0.0,
            'gender_confidence': 0.0,
            'age_confidence': 0.0,
            'race_confidence': 0.0
        }
        
        try:
            # Получаем ключевые точки
            face_oval = features.get('face_oval', np.array([]))
            
            # Пытаемся использовать продвинутый детектор пола (наивысший приоритет - ансамбль методов)
            advanced_gender_result = None
            if self.advanced_gender_detector is not None and self.advanced_gender_detector.is_available():
                try:
                    advanced_result = self.advanced_gender_detector.predict_gender_ensemble(image, face_oval)
                    
                    if advanced_result.get('gender') != 'Не определен' and advanced_result.get('confidence', 0) > 0.3:
                        advanced_gender_result = {
                            'gender': advanced_result['gender'],
                            'confidence': advanced_result['confidence']
                        }
                        logger.info(f"Продвинутый детектор определил пол: {advanced_result['gender']} "
                                   f"(уверенность: {advanced_result['confidence']:.2f}, "
                                   f"голосов: {advanced_result.get('votes', {})})")
                except Exception as e:
                    logger.warning(f"Ошибка при использовании продвинутого детектора пола: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
            
            # Пытаемся использовать DeepFace для определения пола и возраста (резервный метод)
            deepface_gender_result = None
            deepface_age_result = None
            if self.deepface_analyzer is not None and self.deepface_analyzer.is_available() and len(face_oval) > 0:
                try:
                    deepface_result = self.deepface_analyzer.predict_gender_age(image, face_oval)
                    
                    # Проверяем качество результатов DeepFace
                    deepface_gender_conf = deepface_result.get('gender_confidence', 0.0)
                    deepface_age_conf = deepface_result.get('age_confidence', 0.0)
                    deepface_gender = deepface_result.get('gender', 'Не определен')
                    
                    logger.info(f"DeepFace результаты - Пол: {deepface_gender} "
                               f"(уверенность: {deepface_gender_conf:.2f}), "
                               f"Возраст: {deepface_result.get('age', 'Не определен')} "
                               f"(уверенность: {deepface_age_conf:.2f})")
                    
                    # Сохраняем результаты DeepFace
                    if deepface_gender != 'Не определен' and deepface_gender_conf >= 0.5:
                        deepface_gender_result = {
                            'gender': deepface_gender,
                            'confidence': deepface_gender_conf
                        }
                        logger.info(f"DeepFace определил пол: {deepface_gender} (уверенность: {deepface_gender_conf:.2f})")
                    
                    if deepface_result.get('age', 'Не определен') != 'Не определен' and deepface_age_conf >= 0.5:
                        deepface_age_result = {
                            'age': deepface_result['age'],
                            'confidence': deepface_age_conf
                        }
                        logger.info(f"DeepFace определил возраст: {deepface_result['age']} (уверенность: {deepface_age_conf:.2f})")
                except Exception as e:
                    logger.warning(f"Ошибка при использовании DeepFace: {e}. Переход на OpenCV DNN или эвристический метод.")
                    import traceback
                    logger.warning(traceback.format_exc())
            
            # Пытаемся использовать OpenCV DNN для определения пола и возраста (резервный метод)
            dnn_gender_result = None
            dnn_age_result = None
            if self.dnn_analyzer is not None and self.dnn_analyzer.is_available() and len(face_oval) > 0:
                try:
                    dnn_result = self.dnn_analyzer.predict_gender_age(image, face_oval)
                    
                    # Проверяем качество результатов DNN
                    dnn_gender_conf = dnn_result.get('gender_confidence', 0.0)
                    dnn_age_conf = dnn_result.get('age_confidence', 0.0)
                    dnn_gender = dnn_result.get('gender', 'Не определен')
                    
                    logger.info(f"DNN результаты - Пол: {dnn_gender} "
                               f"(уверенность: {dnn_gender_conf:.2f}), "
                               f"Возраст: {dnn_result.get('age', 'Не определен')} "
                               f"(уверенность: {dnn_age_conf:.2f})")
                    
                    # Сохраняем результаты DNN для комбинирования с эвристическим методом
                    if dnn_gender != 'Не определен' and dnn_gender_conf >= 0.4:  # Повышен порог до 0.4
                        dnn_gender_result = {
                            'gender': dnn_gender,
                            'confidence': dnn_gender_conf
                        }
                        logger.info(f"DNN определил пол: {dnn_gender} (уверенность: {dnn_gender_conf:.2f})")
                    
                    if dnn_result.get('age', 'Не определен') != 'Не определен' and dnn_age_conf >= 0.4:
                        dnn_age_result = {
                            'age': dnn_result['age'],
                            'confidence': dnn_age_conf
                        }
                        logger.info(f"DNN определил возраст: {dnn_result['age']} (уверенность: {dnn_age_conf:.2f})")
                except Exception as e:
                    logger.warning(f"Ошибка при использовании OpenCV DNN: {e}. Переход на эвристический метод.")
                    import traceback
                    logger.warning(traceback.format_exc())
            
            # Всегда используем эвристический метод для сравнения и комбинирования (резервный)
            heuristic_result = self._estimate_gender_age_heuristic(features, image)
            
            # Определяем пол с приоритетом: Продвинутый детектор > DeepFace > DNN > эвристика
            # Если метод вернул "Не определен", пробуем следующий метод
            if advanced_gender_result is not None and advanced_gender_result['gender'] != 'Не определен':
                # Продвинутый детектор имеет наивысший приоритет - используем его результат
                result['gender'] = advanced_gender_result['gender']
                result['gender_confidence'] = advanced_gender_result['confidence']
                logger.info(f"Использован продвинутый детектор (ансамбль методов): {result['gender']} "
                           f"(уверенность: {result['gender_confidence']:.2f})")
            elif deepface_gender_result is not None and deepface_gender_result['gender'] != 'Не определен':
                # DeepFace имеет наивысший приоритет - используем его результат
                result['gender'] = deepface_gender_result['gender']
                result['gender_confidence'] = deepface_gender_result['confidence']
                logger.info(f"Использован DeepFace (приоритетный метод): {result['gender']} "
                           f"(уверенность: {result['gender_confidence']:.2f})")
            elif dnn_gender_result is not None and dnn_gender_result['gender'] != 'Не определен' and heuristic_result.get('gender') != 'Не определен':
                # Комбинируем результаты DNN и эвристического метода для пола
                # Если оба метода согласны, используем взвешенное среднее уверенности
                if dnn_gender_result['gender'] == heuristic_result['gender']:
                    # Оба метода согласны - используем взвешенное среднее
                    dnn_weight = min(0.6, dnn_gender_result['confidence'])  # Максимальный вес 0.6
                    heuristic_weight = 1.0 - dnn_weight
                    result['gender'] = dnn_gender_result['gender']
                    result['gender_confidence'] = (
                        dnn_gender_result['confidence'] * dnn_weight + 
                        heuristic_result['gender_confidence'] * heuristic_weight
                    )
                    logger.info(f"Комбинированный результат (согласны): {result['gender']} "
                               f"(уверенность: {result['gender_confidence']:.2f})")
                else:
                    # Методы не согласны - приоритет DNN, если он очень уверен (>=0.95)
                    heuristic_conf = heuristic_result['gender_confidence']
                    dnn_conf = dnn_gender_result['confidence']
                    
                    # Если DNN очень уверен (>=0.95), доверяем ему больше, чем эвристике
                    if dnn_conf >= 0.95:
                        result['gender'] = dnn_gender_result['gender']
                        # Снижаем уверенность немного из-за конфликта, но доверяем DNN
                        result['gender_confidence'] = min(0.90, dnn_conf * 0.95)
                        logger.warning(f"Конфликт методов: выбран DNN {result['gender']} "
                                      f"(уверенность DNN: {dnn_conf:.2f}, "
                                      f"эвристика: {heuristic_result['gender']} {heuristic_conf:.2f}). "
                                      f"DNN имеет приоритет при очень высокой уверенности (>=0.95).")
                    # Если эвристика очень уверена (>=0.85) и DNN не очень уверен (<0.9)
                    elif heuristic_conf >= 0.85 and dnn_conf < 0.9:
                        result['gender'] = heuristic_result['gender']
                        result['gender_confidence'] = min(0.85, heuristic_conf * 0.9)
                        logger.warning(f"Конфликт методов: выбрана эвристика {result['gender']} "
                                      f"(уверенность эвристики: {heuristic_conf:.2f}, "
                                      f"DNN: {dnn_gender_result['gender']} {dnn_conf:.2f}). "
                                      f"Эвристика имеет приоритет при очень высокой уверенности.")
                    # Если DNN уверен (>=0.85), но эвристика тоже уверена
                    elif dnn_conf >= 0.85 and heuristic_conf >= 0.6:
                        # Выбираем DNN, так как он более надежен при высокой уверенности
                        result['gender'] = dnn_gender_result['gender']
                        result['gender_confidence'] = min(0.80, dnn_conf * 0.9)
                        logger.warning(f"Конфликт методов: выбран DNN {result['gender']} "
                                      f"(уверенность DNN: {dnn_conf:.2f}, "
                                      f"эвристика: {heuristic_result['gender']} {heuristic_conf:.2f}). "
                                      f"DNN имеет приоритет при высокой уверенности.")
                    # Если эвристика не уверена, но DNN уверен
                    elif dnn_conf >= 0.7 and heuristic_conf < 0.6:
                        result['gender'] = dnn_gender_result['gender']
                        result['gender_confidence'] = dnn_conf * 0.85
                        logger.warning(f"Конфликт методов: выбран DNN {result['gender']} "
                                      f"(уверенность DNN: {dnn_conf:.2f}, "
                                      f"эвристика: {heuristic_result['gender']} {heuristic_conf:.2f}). "
                                      f"DNN выбран, так как эвристика не уверена.")
                    else:
                        # Выбираем более уверенный метод, но снижаем уверенность
                        if dnn_conf > heuristic_conf:
                            result['gender'] = dnn_gender_result['gender']
                            result['gender_confidence'] = dnn_conf * 0.75
                        else:
                            result['gender'] = heuristic_result['gender']
                            result['gender_confidence'] = heuristic_conf * 0.75
                        logger.warning(f"Конфликт методов: выбран более уверенный метод {result['gender']} "
                                      f"(DNN: {dnn_gender_result['gender']} {dnn_conf:.2f}, "
                                      f"эвристика: {heuristic_result['gender']} {heuristic_conf:.2f})")
            elif dnn_gender_result is not None and dnn_gender_result['gender'] != 'Не определен':
                # Только DNN дал результат - используем его, но с осторожностью
                result['gender'] = dnn_gender_result['gender']
                # Если уверенность очень высокая (>0.9), немного снижаем для безопасности
                if dnn_gender_result['confidence'] > 0.9:
                    result['gender_confidence'] = dnn_gender_result['confidence'] * 0.85
                    logger.warning(f"Использован только DNN с очень высокой уверенностью: {result['gender']} "
                                  f"(уверенность снижена с {dnn_gender_result['confidence']:.2f} до {result['gender_confidence']:.2f})")
                else:
                    result['gender_confidence'] = dnn_gender_result['confidence']
                logger.info(f"Использован только DNN: {result['gender']} (уверенность: {result['gender_confidence']:.2f})")
            else:
                # Используем эвристический метод или возвращаем "Не определен", если и он не определил
                if heuristic_result.get('gender') != 'Не определен':
                    result['gender'] = heuristic_result['gender']
                    result['gender_confidence'] = heuristic_result['gender_confidence']
                    logger.info(f"Использован только эвристический метод: {result['gender']} (уверенность: {result['gender_confidence']:.2f})")
                else:
                    # Все методы вернули "Не определен"
                    result['gender'] = 'Не определен'
                    result['gender_confidence'] = 0.0
                    logger.warning("Все методы определения пола вернули 'Не определен'. Пол не может быть определен.")
            
            # Для возраста используем приоритет: DeepFace > DNN > эвристика
            if deepface_age_result is not None:
                result['age'] = deepface_age_result['age']
                result['age_confidence'] = deepface_age_result['confidence']
                logger.info(f"Использован DeepFace для возраста: {result['age']} "
                           f"(уверенность: {result['age_confidence']:.2f})")
            elif dnn_age_result is not None:
                result['age'] = dnn_age_result['age']
                result['age_confidence'] = dnn_age_result['confidence']
            else:
                result['age'] = heuristic_result['age']
                result['age_confidence'] = heuristic_result['age_confidence']
            
            # Определяем расу с приоритетом: RaceAnalyzer (DeepFace/InsightFace) > эвристика
            race_result = None
            if self.race_analyzer is not None and self.race_analyzer.is_available() and len(face_oval) > 0:
                try:
                    race_result = self.race_analyzer.predict_race(image, face_oval)
                    if race_result.get('race') != 'Не определена' and race_result.get('race_confidence', 0) > 0.3:
                        result['race'] = race_result['race']
                        result['race_confidence'] = race_result['race_confidence']
                        logger.info(f"Анализатор расы определил: {result['race']} "
                                   f"(уверенность: {result['race_confidence']:.2f}, "
                                   f"метод: {race_result.get('method', 'unknown')})")
                    else:
                        # Если результат неудовлетворительный, используем эвристику
                        race_result = None
                except Exception as e:
                    logger.warning(f"Ошибка при использовании анализатора расы: {e}. Переход на эвристический метод.")
                    import traceback
                    logger.warning(traceback.format_exc())
                    race_result = None
            
            # Если анализатор расы не дал результат, используем эвристический метод
            if race_result is None or race_result.get('race') == 'Не определена':
                race_result = self._estimate_race_heuristic(features, image)
                result['race'] = race_result['race']
                result['race_confidence'] = race_result['race_confidence']
                logger.info(f"Использован эвристический метод для расы: {result['race']} "
                           f"(уверенность: {result['race_confidence']:.2f})")
            
            # Вычисляем общую уверенность
            result['confidence'] = (result['gender_confidence'] + 
                                 result['age_confidence'] + 
                                 result['race_confidence']) / 3.0
            
            return result
            
        except Exception as e:
            # В случае ошибки возвращаем значения по умолчанию
            import traceback
            traceback.print_exc()
            return result
    
    def _estimate_gender_age_heuristic(self, features: Dict, image: np.ndarray) -> Dict[str, any]:
        """
        Эвристический метод определения пола, возраста и расы (резервный метод)
        
        Args:
            features: Словарь с характеристиками лица
            image: Изображение лица
            
        Returns:
            Словарь с оценками пола, возраста и расы
        """
        result = {
            'gender': 'Не определен', 
            'age': 'Не определен', 
            'race': 'Не определена',
            'confidence': 0.0,
            'gender_confidence': 0.0,
            'age_confidence': 0.0,
            'race_confidence': 0.0
        }
        
        try:
            # Получаем ключевые точки
            face_oval = features.get('face_oval', np.array([]))
            left_eye = features.get('left_eye', np.array([]))
            right_eye = features.get('right_eye', np.array([]))
            nose_tip = features.get('nose_tip', np.array([]))
            nose_bridge = features.get('nose_bridge', np.array([]))
            mouth_outer = features.get('mouth_outer', np.array([]))
            chin = features.get('chin', np.array([]))
            forehead = features.get('forehead', np.array([]))
            left_eyebrow = features.get('left_eyebrow', np.array([]))
            right_eyebrow = features.get('right_eyebrow', np.array([]))
            left_cheek = features.get('left_cheek', np.array([]))
            right_cheek = features.get('right_cheek', np.array([]))
            
            if len(face_oval) == 0:
                return result
            
            # Вычисляем пропорции лица
            face_width = float(np.max(face_oval[:, 0]) - np.min(face_oval[:, 0]))
            face_height = float(np.max(face_oval[:, 1]) - np.min(face_oval[:, 1]))
            
            if face_height == 0:
                return result
            
            # Соотношение ширины к высоте лица
            face_ratio = float(face_width / face_height)
            
            # ========== УЛУЧШЕННОЕ ОПРЕДЕЛЕНИЕ ПОЛА ==========
            # Используем взвешенную систему оценок для более точного определения
            # Улучшенные пороги на основе современных исследований антропометрии лица
            male_indicators = 0.0
            female_indicators = 0.0
            total_weight = 0.0
            
            # 1. Общая форма лица (вес: 1.2 - увеличен для лучшей точности)
            # Исправленные пороги на основе статистики различных этнических групп
            # Мужские лица обычно шире (face_ratio > 0.75), женские уже (face_ratio < 0.72)
            if face_ratio > 0.78:  # Более широкое лицо (исправлен порог)
                male_indicators += 1.2
                total_weight += 1.2
            elif face_ratio < 0.72:  # Более узкое лицо (исправлен порог)
                female_indicators += 1.2
                total_weight += 1.2
            elif 0.72 <= face_ratio <= 0.78:  # Среднее (исправлен диапазон)
                # Градиентное влияние в зависимости от близости к границам
                if face_ratio > 0.75:
                    male_indicators += 0.4 * (face_ratio - 0.72) / 0.06
                else:
                    female_indicators += 0.4 * (0.75 - face_ratio) / 0.03
                total_weight += 0.4
            
            # 2. Анализ формы подбородка (вес: 1.3 - увеличен для лучшей точности)
            if len(chin) > 0:
                chin_width = float(np.max(chin[:, 0]) - np.min(chin[:, 0]))
                chin_height = float(np.max(chin[:, 1]) - np.min(chin[:, 1]))
                chin_ratio = float(chin_width / face_width) if face_width > 0 else 0.0
                chin_aspect = float(chin_width / chin_height) if chin_height > 0 else 0.0
                
                # Исправленные пороги на основе антропометрических данных
                # Мужские подбородки обычно шире относительно лица
                if chin_ratio > 0.42:  # Широкий подбородок (исправлен порог)
                    male_indicators += 0.65
                    total_weight += 0.65
                elif chin_ratio < 0.35:  # Узкий подбородок (исправлен порог)
                    female_indicators += 0.65
                    total_weight += 0.65
                else:
                    # Градиентное влияние для промежуточных значений
                    if chin_ratio > 0.38:
                        male_indicators += 0.3 * (chin_ratio - 0.35) / 0.07
                    else:
                        female_indicators += 0.3 * (0.38 - chin_ratio) / 0.03
                    total_weight += 0.3
                
                # Мужские подбородки обычно более квадратные
                if chin_aspect > 1.45:  # Квадратный подбородок (исправлен порог)
                    male_indicators += 0.65
                    total_weight += 0.65
                elif chin_aspect < 1.15:  # Округлый подбородок (исправлен порог)
                    female_indicators += 0.65
                    total_weight += 0.65
                else:
                    # Градиентное влияние
                    if chin_aspect > 1.30:
                        male_indicators += 0.3 * (chin_aspect - 1.15) / 0.30
                    else:
                        female_indicators += 0.3 * (1.30 - chin_aspect) / 0.15
                    total_weight += 0.3
            
            # 3. Анализ размера и формы глаз (вес: 1.4 - увеличен для лучшего определения женских лиц)
            if len(left_eye) > 0 and len(right_eye) > 0:
                eye_size1 = float(np.max(left_eye[:, 0]) - np.min(left_eye[:, 0]))
                eye_size2 = float(np.max(right_eye[:, 0]) - np.min(right_eye[:, 0]))
                eye_height1 = float(np.max(left_eye[:, 1]) - np.min(left_eye[:, 1]))
                eye_height2 = float(np.max(right_eye[:, 1]) - np.min(right_eye[:, 1]))
                
                avg_eye_width = float((eye_size1 + eye_size2) / 2)
                avg_eye_height = float((eye_height1 + eye_height2) / 2)
                eye_ratio = float(avg_eye_width / face_width) if face_width > 0 else 0.0
                eye_aspect = float(avg_eye_width / avg_eye_height) if avg_eye_height > 0 else 0.0
                
                # Исправленные пороги на основе исследований пропорций лица
                # Женские глаза обычно больше относительно лица
                if eye_ratio > 0.14:  # Большие глаза (исправлен порог)
                    female_indicators += 0.7
                    total_weight += 0.7
                elif eye_ratio < 0.10:  # Маленькие глаза (исправлен порог)
                    male_indicators += 0.7
                    total_weight += 0.7
                else:
                    # Градиентное влияние
                    if eye_ratio > 0.12:
                        female_indicators += 0.35 * (eye_ratio - 0.10) / 0.04
                    else:
                        male_indicators += 0.35 * (0.12 - eye_ratio) / 0.02
                    total_weight += 0.35
                
                # Женские глаза обычно шире (больше соотношение ширина/высота)
                if eye_aspect > 2.6:  # Широкие глаза (исправлен порог)
                    female_indicators += 0.7
                    total_weight += 0.7
                elif eye_aspect < 2.1:  # Узкие глаза (исправлен порог)
                    male_indicators += 0.7
                    total_weight += 0.7
                else:
                    # Градиентное влияние
                    if eye_aspect > 2.35:
                        female_indicators += 0.35 * (eye_aspect - 2.1) / 0.50
                    else:
                        male_indicators += 0.35 * (2.35 - eye_aspect) / 0.25
                    total_weight += 0.35
            
            # 4. Анализ бровей (вес: 0.8)
            if len(left_eyebrow) > 0 and len(right_eyebrow) > 0:
                brow_thickness1 = float(np.max(left_eyebrow[:, 1]) - np.min(left_eyebrow[:, 1]))
                brow_thickness2 = float(np.max(right_eyebrow[:, 1]) - np.min(right_eyebrow[:, 1]))
                avg_brow_thickness = float((brow_thickness1 + brow_thickness2) / 2)
                brow_ratio = float(avg_brow_thickness / face_height) if face_height > 0 else 0.0
                
                if brow_ratio > 0.016:  # Толстые брови
                    male_indicators += 0.8
                    total_weight += 0.8
                elif brow_ratio < 0.012:  # Тонкие брови
                    female_indicators += 0.8
                    total_weight += 0.8
            
            # 5. Анализ носа (вес: 1.0)
            if len(nose_tip) > 0 and len(nose_bridge) > 0:
                nose_width = float(np.max(nose_tip[:, 0]) - np.min(nose_tip[:, 0]))
                nose_height = float(np.max(nose_tip[:, 1]) - np.min(nose_tip[:, 1]))
                nose_ratio = float(nose_width / face_width) if face_width > 0 else 0.0
                nose_height_ratio = float(nose_height / face_height) if face_height > 0 else 0.0
                
                if nose_ratio > 0.17:  # Широкий нос
                    male_indicators += 0.5
                    total_weight += 0.5
                elif nose_ratio < 0.13:  # Узкий нос
                    female_indicators += 0.5
                    total_weight += 0.5
                
                if nose_height_ratio > 0.27:  # Длинный нос
                    male_indicators += 0.5
                    total_weight += 0.5
                elif nose_height_ratio < 0.22:  # Короткий нос
                    female_indicators += 0.5
                    total_weight += 0.5
            
            # 6. Анализ губ (вес: 0.6)
            if len(mouth_outer) > 0:
                mouth_width = float(np.max(mouth_outer[:, 0]) - np.min(mouth_outer[:, 0]))
                mouth_height = float(np.max(mouth_outer[:, 1]) - np.min(mouth_outer[:, 1]))
                mouth_ratio = float(mouth_width / face_width) if face_width > 0 else 0.0
                mouth_aspect = float(mouth_width / mouth_height) if mouth_height > 0 else 0.0
                
                if mouth_ratio > 0.48:  # Широкий рот
                    female_indicators += 0.3
                    total_weight += 0.3
                elif mouth_ratio < 0.40:  # Узкий рот
                    male_indicators += 0.3
                    total_weight += 0.3
                
                if mouth_aspect > 3.3:  # Широкие губы
                    female_indicators += 0.3
                    total_weight += 0.3
            
            # 7. Анализ скул (вес: 0.5)
            if len(left_cheek) > 0 and len(right_cheek) > 0:
                cheek1_center_y = float(np.mean(left_cheek[:, 1]))
                cheek2_center_y = float(np.mean(right_cheek[:, 1]))
                face_center_y = float(np.mean(face_oval[:, 1]))
                cheek_position = float((cheek1_center_y + cheek2_center_y) / 2)
                
                if cheek_position < face_center_y - face_height * 0.08:  # Высокие скулы
                    female_indicators += 0.5
                    total_weight += 0.5
                elif cheek_position > face_center_y + face_height * 0.05:  # Низкие скулы
                    male_indicators += 0.5
                    total_weight += 0.5
            
            # Вычисляем итоговую оценку с улучшенной логикой
            if total_weight > 0:
                male_score = male_indicators / total_weight
                female_score = female_indicators / total_weight
                difference = abs(male_score - female_score)
                
                # Улучшенная логика определения пола с учетом качества данных
                # Если разница значительная (>0.15), используем более высокую уверенность
                if male_score > female_score and difference > 0.15:  # Четкое преобладание мужских признаков
                    result['gender'] = 'Мужской'
                    # Улучшенная формула уверенности с учетом качества признаков
                    result['gender_confidence'] = min(0.95, 0.60 + difference * 0.7)
                elif female_score > male_score and difference > 0.15:  # Четкое преобладание женских признаков
                    result['gender'] = 'Женский'
                    result['gender_confidence'] = min(0.95, 0.60 + difference * 0.7)
                elif difference > 0.08:  # Умеренное преобладание
                    if male_score > female_score:
                        result['gender'] = 'Мужской'
                        result['gender_confidence'] = 0.55 + difference * 0.5
                    else:
                        result['gender'] = 'Женский'
                        result['gender_confidence'] = 0.55 + difference * 0.5
                else:  # Неопределенный случай
                    # Выбираем по небольшому перевесу, но с низкой уверенностью
                    if male_score > female_score:
                        result['gender'] = 'Мужской'
                        result['gender_confidence'] = 0.48 + difference * 0.4
                    elif female_score > male_score:
                        result['gender'] = 'Женский'
                        result['gender_confidence'] = 0.48 + difference * 0.4
                    else:
                        result['gender'] = 'Не определен'
                        result['gender_confidence'] = 0.3
            else:
                result['gender'] = 'Не определен'
                result['gender_confidence'] = 0.3
            
            # ========== УЛУЧШЕННОЕ ОПРЕДЕЛЕНИЕ ВОЗРАСТА ==========
            age_score = 0
            age_features_count = 0
            
            # 1. Соотношение высоты лба к высоте лица
            if len(forehead) > 0 and len(chin) > 0:
                forehead_y = float(np.min(forehead[:, 1]))
                chin_y = float(np.max(chin[:, 1]))
                face_height_actual = float(chin_y - forehead_y)
                
                if face_height_actual > 0:
                    forehead_height = float(np.max(forehead[:, 1]) - np.min(forehead[:, 1]))
                    forehead_ratio = float(forehead_height / face_height_actual)
                    
                    if forehead_ratio > 0.38:
                        age_score -= 12  # Большой лоб - моложе
                    elif forehead_ratio < 0.30:
                        age_score += 8  # Меньший лоб - старше
                    age_features_count += 1
            
            # 2. Анализ размера и формы носа (с возрастом нос растет)
            if len(nose_tip) > 0:
                nose_size = float(np.max(nose_tip[:, 1]) - np.min(nose_tip[:, 1]))
                nose_ratio = float(nose_size / face_height) if face_height > 0 else 0.0
                
                if nose_ratio > 0.28:
                    age_score += 8  # Большой нос - старше
                elif nose_ratio < 0.20:
                    age_score -= 6  # Меньший нос - моложе
                age_features_count += 1
            
            # 3. Анализ расстояния между глазами и ртом (увеличивается с возрастом)
            if len(left_eye) > 0 and len(right_eye) > 0 and len(mouth_outer) > 0:
                eye_center_y = float((np.mean(left_eye[:, 1]) + np.mean(right_eye[:, 1])) / 2)
                mouth_center_y = float(np.mean(mouth_outer[:, 1]))
                eye_mouth_distance = float(mouth_center_y - eye_center_y)
                distance_ratio = float(eye_mouth_distance / face_height) if face_height > 0 else 0.0
                
                if distance_ratio > 0.35:
                    age_score += 6  # Большое расстояние - старше
                elif distance_ratio < 0.28:
                    age_score -= 5  # Маленькое расстояние - моложе
                age_features_count += 1
            
            # 4. Анализ размера глаз (уменьшаются с возрастом)
            if len(left_eye) > 0 and len(right_eye) > 0:
                eye_size1 = float(np.max(left_eye[:, 0]) - np.min(left_eye[:, 0]))
                eye_size2 = float(np.max(right_eye[:, 0]) - np.min(right_eye[:, 0]))
                avg_eye_size = float((eye_size1 + eye_size2) / 2)
                eye_ratio = float(avg_eye_size / face_width) if face_width > 0 else 0.0
                
                if eye_ratio < 0.12:
                    age_score += 5  # Маленькие глаза - старше
                elif eye_ratio > 0.16:
                    age_score -= 4  # Большие глаза - моложе
                age_features_count += 1
            
            # 5. Анализ формы лица (становится длиннее с возрастом)
            if face_ratio < 0.70:
                age_score += 4  # Длинное лицо - старше
            elif face_ratio > 0.80:
                age_score -= 3  # Широкое лицо - моложе
            age_features_count += 1
            
            # 6. Анализ текстуры кожи (морщины) - упрощенный метод
            try:
                # Извлекаем область лица для анализа текстуры
                x_min, y_min = int(np.min(face_oval[:, 0])), int(np.min(face_oval[:, 1]))
                x_max, y_max = int(np.max(face_oval[:, 0])), int(np.max(face_oval[:, 1]))
                
                # Добавляем отступы
                padding = 10
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(image.shape[1], x_max + padding)
                y_max = min(image.shape[0], y_max + padding)
                
                face_roi = image[y_min:y_max, x_min:x_max]
                if face_roi.size > 0:
                    # Конвертируем в grayscale
                    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
                    
                    # Применяем фильтр Лапласа для обнаружения краев (морщин)
                    # Используем cv2.CV_64F для совместимости с OpenCV 4.x
                    # В OpenCV 4.x можно также использовать cv2.CV_64F или просто указать dtype
                    laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
                    variance = float(np.var(laplacian))
                    
                    # Высокая вариация указывает на больше морщин (старший возраст)
                    if variance > 500:
                        age_score += 7
                    elif variance < 200:
                        age_score -= 5
                    age_features_count += 1
            except:
                pass
            
            # Нормализуем оценку возраста с улучшенной формулой
            if age_features_count > 0:
                # Используем взвешенное среднее с учетом важности признаков
                age_score = age_score / age_features_count
                # Применяем нелинейную нормализацию для лучшей точности
                age_score = np.sign(age_score) * (abs(age_score) ** 0.9)
            
            # Базовый возраст (примерно 30 лет) + корректировка
            # Скорректирован базовый возраст для более точных результатов
            estimated_age = 30 + int(age_score * 1.1)  # Небольшое усиление влияния признаков
            
            # Ограничиваем разумными пределами
            estimated_age = max(16, min(85, estimated_age))
            
            # Вычисляем уверенность в возрасте с улучшенной формулой
            # Учитываем количество использованных признаков
            base_confidence = 0.52 + abs(age_score) / 18.0
            # Бонус за использование большего количества признаков
            if age_features_count >= 5:
                base_confidence += 0.08
            elif age_features_count >= 3:
                base_confidence += 0.04
            
            age_confidence = min(0.88, base_confidence)
            result['age_confidence'] = age_confidence
            
            # Форматируем возраст с учетом уверенности (улучшенные диапазоны)
            if age_confidence > 0.75:
                result['age'] = f"{estimated_age} ± 3 лет"
            elif age_confidence > 0.60:
                result['age'] = f"{estimated_age} ± 5 лет"
            elif age_confidence > 0.50:
                result['age'] = f"{estimated_age} ± 7 лет"
            else:
                result['age'] = f"{estimated_age} ± 9 лет"
            
            # Определяем расу эвристическим методом
            race_result = self._estimate_race_heuristic(features, image)
            result['race'] = race_result['race']
            result['race_confidence'] = race_result['race_confidence']
            
            # Общая уверенность
            result['confidence'] = (result['gender_confidence'] + result['age_confidence'] + result['race_confidence']) / 3.0
            
        except Exception as e:
            # В случае ошибки возвращаем значения по умолчанию
            import traceback
            traceback.print_exc()
        
        return result
    
    def _estimate_race_heuristic(self, features: Dict, image: np.ndarray) -> Dict[str, any]:
        """
        Эвристический метод определения расы
        
        Args:
            features: Словарь с характеристиками лица
            image: Изображение лица
            
        Returns:
            Словарь с оценкой расы
        """
        result = {
            'race': 'Не определена',
            'race_confidence': 0.0
        }
        
        try:
            # Получаем ключевые точки
            face_oval = features.get('face_oval', np.array([]))
            left_eye = features.get('left_eye', np.array([]))
            right_eye = features.get('right_eye', np.array([]))
            nose_tip = features.get('nose_tip', np.array([]))
            nose_bridge = features.get('nose_bridge', np.array([]))
            left_cheek = features.get('left_cheek', np.array([]))
            
            if len(face_oval) == 0:
                return result
            
            # Вычисляем пропорции лица
            face_width = float(np.max(face_oval[:, 0]) - np.min(face_oval[:, 0]))
            face_height = float(np.max(face_oval[:, 1]) - np.min(face_oval[:, 1]))
            
            if face_height == 0:
                return result
            
            face_ratio = float(face_width / face_height)
            
            # ========== ОПРЕДЕЛЕНИЕ РАСЫ ==========
            race_score = {'asian': 0.0, 'caucasian': 0.0, 'african': 0.0, 'hispanic': 0.0}
            race_features_count = 0
            
            # 1. Анализ цвета кожи (основной индикатор)
            try:
                # Извлекаем область щек для анализа цвета
                if len(left_cheek) > 0:
                    x_min = int(np.min(left_cheek[:, 0]))
                    y_min = int(np.min(left_cheek[:, 1]))
                    x_max = int(np.max(left_cheek[:, 0]))
                    y_max = int(np.max(left_cheek[:, 1]))
                    
                    if x_max > x_min and y_max > y_min:
                        cheek_roi = image[y_min:y_max, x_min:x_max]
                        if cheek_roi.size > 0:
                            # Конвертируем в HSV для анализа цвета
                            hsv_roi = cv2.cvtColor(cheek_roi, cv2.COLOR_BGR2HSV)
                            mean_hue = float(np.mean(hsv_roi[:, :, 0]))
                            mean_saturation = float(np.mean(hsv_roi[:, :, 1]))
                            mean_value = float(np.mean(hsv_roi[:, :, 2]))
                            
                            # Анализ тона кожи
                            if mean_value < 80:  # Темная кожа
                                race_score['african'] += 0.6
                            elif mean_value > 180:  # Светлая кожа
                                race_score['caucasian'] += 0.5
                                if mean_hue < 20 or mean_hue > 160:  # Розоватый оттенок
                                    race_score['caucasian'] += 0.2
                            elif 100 < mean_value < 150:  # Средний тон
                                race_score['hispanic'] += 0.4
                                race_score['asian'] += 0.3
                            
                            # Азиатская кожа часто имеет желтоватый оттенок
                            if 15 < mean_hue < 35 and mean_saturation > 50:
                                race_score['asian'] += 0.4
                            
                            race_features_count += 1
            except:
                pass
            
            # 2. Геометрические признаки лица
            # Азиатские лица: более плоское лицо, меньший нос
            if len(nose_bridge) > 0:
                nose_bridge_height = float(np.max(nose_bridge[:, 1]) - np.min(nose_bridge[:, 1]))
                nose_bridge_ratio = float(nose_bridge_height / face_height) if face_height > 0 else 0.0
                
                if nose_bridge_ratio < 0.12:
                    race_score['asian'] += 0.3
                elif nose_bridge_ratio > 0.18:
                    race_score['caucasian'] += 0.2
                    race_score['african'] += 0.2
                race_features_count += 1
            
            # Ширина носа
            if len(nose_tip) > 0:
                nose_width = float(np.max(nose_tip[:, 0]) - np.min(nose_tip[:, 0]))
                nose_width_ratio = float(nose_width / face_width) if face_width > 0 else 0.0
                
                if nose_width_ratio > 0.20:
                    race_score['african'] += 0.3
                elif nose_width_ratio < 0.14:
                    race_score['asian'] += 0.2
                race_features_count += 1
            
            # Форма глаз (азиатские глаза часто более узкие)
            if len(left_eye) > 0 and len(right_eye) > 0:
                eye_height1 = float(np.max(left_eye[:, 1]) - np.min(left_eye[:, 1]))
                eye_height2 = float(np.max(right_eye[:, 1]) - np.min(right_eye[:, 1]))
                eye_width1 = float(np.max(left_eye[:, 0]) - np.min(left_eye[:, 0]))
                eye_width2 = float(np.max(right_eye[:, 0]) - np.min(right_eye[:, 0]))
                
                avg_eye_height = float((eye_height1 + eye_height2) / 2)
                avg_eye_width = float((eye_width1 + eye_width2) / 2)
                eye_aspect = float(avg_eye_width / avg_eye_height) if avg_eye_height > 0 else 0.0
                
                if eye_aspect > 3.0:
                    race_score['asian'] += 0.2
                race_features_count += 1
            
            # Форма лица
            if face_ratio < 0.70:
                race_score['african'] += 0.15
            elif face_ratio > 0.80:
                race_score['asian'] += 0.15
            race_features_count += 1
            
            # Нормализуем оценки расы
            if race_features_count > 0:
                for race in race_score:
                    race_score[race] = race_score[race] / race_features_count
            
            # Определяем расу
            max_race = max(race_score.items(), key=lambda x: x[1])
            race_names = {
                'asian': 'Азиатская',
                'caucasian': 'Европеоидная',
                'african': 'Негроидная',
                'hispanic': 'Латиноамериканская'
            }
            
            # Снижаем порог для определения расы, чтобы она определялась чаще
            if max_race[1] > 0.15:  # Снижен порог с 0.25 до 0.15
                result['race'] = race_names.get(max_race[0], 'Не определена')
                result['race_confidence'] = min(0.9, max(0.5, max_race[1] * 2.0))  # Увеличена уверенность
            else:
                result['race'] = 'Не определена'
                result['race_confidence'] = 0.3
            
        except Exception as e:
            import traceback
            traceback.print_exc()
        
        return result

