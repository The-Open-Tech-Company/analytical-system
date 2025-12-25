"""
Модуль для определения расы человека с использованием DeepFace и InsightFace
Использует современные модели глубокого обучения для высокой точности
"""
import os
import cv2
import numpy as np
from typing import Dict, Optional
import logging
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импорт DeepFace
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("DeepFace успешно импортирован для определения расы")
except ImportError as e:
    logger.warning(f"DeepFace не доступен для определения расы: {e}")
    DEEPFACE_AVAILABLE = False
except Exception as e:
    logger.warning(f"Ошибка при импорте DeepFace: {e}")
    DEEPFACE_AVAILABLE = False

# Импорт InsightFace
INSIGHTFACE_AVAILABLE = False
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
    logger.info("InsightFace успешно импортирован для определения расы")
except ImportError as e:
    logger.warning(f"InsightFace не доступен: {e}")
    INSIGHTFACE_AVAILABLE = False
except Exception as e:
    logger.warning(f"Ошибка при импорте InsightFace: {e}")
    INSIGHTFACE_AVAILABLE = False


class RaceAnalyzer:
    """Класс для определения расы с использованием DeepFace и InsightFace"""
    
    def __init__(self):
        """Инициализация анализатора расы"""
        self.deepface_available = DEEPFACE_AVAILABLE
        self.insightface_available = False
        self.insightface_app = None
        
        # Инициализируем InsightFace если доступен
        if INSIGHTFACE_AVAILABLE:
            try:
                # Используем модель по умолчанию (buffalo_l)
                # Эта модель поддерживает определение расы
                self.insightface_app = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CPUExecutionProvider']  # Используем CPU, можно добавить CUDA
                )
                self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
                self.insightface_available = True
                logger.info("InsightFace успешно инициализирован для определения расы")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать InsightFace: {e}")
                self.insightface_available = False
                self.insightface_app = None
        
        logger.info(f"RaceAnalyzer инициализирован. "
                   f"DeepFace: {self.deepface_available}, "
                   f"InsightFace: {self.insightface_available}")
    
    def _extract_face_roi(self, image: np.ndarray, face_oval: np.ndarray) -> Optional[np.ndarray]:
        """
        Извлекает область лица из изображения
        
        Args:
            image: Исходное изображение (BGR)
            face_oval: Массив точек овала лица
            
        Returns:
            Область лица или None
        """
        if len(face_oval) == 0:
            return None
        
        # Получаем границы лица
        x_min = int(np.min(face_oval[:, 0]))
        y_min = int(np.min(face_oval[:, 1]))
        x_max = int(np.max(face_oval[:, 0]))
        y_max = int(np.max(face_oval[:, 1]))
        
        # Добавляем отступы (30% с каждой стороны)
        padding_x = int((x_max - x_min) * 0.3)
        padding_y = int((y_max - y_min) * 0.3)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(image.shape[1], x_max + padding_x)
        y_max = min(image.shape[0], y_max + padding_y)
        
        # Извлекаем область лица
        face_roi = image[y_min:y_max, x_min:x_max]
        
        if face_roi.size == 0:
            return None
        
        return face_roi
    
    def _save_temp_image(self, image: np.ndarray) -> Optional[str]:
        """
        Сохраняет изображение во временный файл
        
        Args:
            image: Изображение (BGR)
            
        Returns:
            Путь к временному файлу или None
        """
        try:
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"race_analyzer_{os.getpid()}.jpg")
            
            # Конвертируем BGR в RGB для сохранения
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Сохраняем изображение
            from PIL import Image
            pil_image = Image.fromarray(rgb_image)
            pil_image.save(temp_path, "JPEG", quality=95)
            
            return temp_path
        except Exception as e:
            logger.error(f"Ошибка при сохранении временного изображения: {e}")
            return None
    
    def _predict_race_deepface(self, image: np.ndarray, face_oval: np.ndarray) -> Optional[Dict]:
        """
        Определяет расу с использованием DeepFace
        
        Args:
            image: Изображение (BGR)
            face_oval: Массив точек овала лица
            
        Returns:
            Словарь с результатами или None
        """
        if not self.deepface_available:
            return None
        
        # Извлекаем область лица
        face_roi = self._extract_face_roi(image, face_oval)
        if face_roi is None:
            return None
        
        # Проверяем размер области лица
        if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
            return None
        
        temp_path = None
        try:
            # Сохраняем изображение во временный файл
            temp_path = self._save_temp_image(face_roi)
            if temp_path is None:
                return None
            
            # Используем DeepFace для анализа расы
            try:
                analysis = DeepFace.analyze(
                    img_path=temp_path,
                    actions=['race'],
                    enforce_detection=False,
                    silent=True
                )
                
                # DeepFace возвращает список или словарь
                if isinstance(analysis, list):
                    analysis = analysis[0]
                
                # Извлекаем результаты расы
                race_result = analysis.get('dominant_race', '') or analysis.get('race', '')
                race_probs = analysis.get('race', {})
                
                # Если race_result - это словарь с вероятностями
                if isinstance(race_result, dict):
                    if race_result:
                        race_result = max(race_result, key=race_result.get)
                        race_probs = race_result
                    else:
                        return None
                
                # Преобразуем название расы в русский формат
                race_mapping = {
                    'asian': 'Азиатская',
                    'indian': 'Индийская',
                    'black': 'Негроидная',
                    'white': 'Европеоидная',
                    'middle eastern': 'Ближневосточная',
                    'latino hispanic': 'Латиноамериканская',
                    'middle eastern': 'Ближневосточная',
                    'latino': 'Латиноамериканская',
                    'hispanic': 'Латиноамериканская'
                }
                
                race_str = str(race_result).lower()
                race_ru = None
                
                # Ищем соответствие
                for key, value in race_mapping.items():
                    if key in race_str:
                        race_ru = value
                        break
                
                if race_ru is None:
                    # Если не нашли точное соответствие, пробуем извлечь из вероятностей
                    if isinstance(race_probs, dict):
                        # Находим расу с максимальной вероятностью
                        max_race = max(race_probs.items(), key=lambda x: x[1])
                        race_key = max_race[0].lower()
                        for key, value in race_mapping.items():
                            if key in race_key:
                                race_ru = value
                                break
                
                if race_ru is None:
                    return None
                
                # Получаем уверенность
                if isinstance(race_probs, dict):
                    # Находим максимальную вероятность
                    max_prob = max(race_probs.values())
                    confidence = float(max_prob)
                else:
                    confidence = 0.75  # По умолчанию
                
                return {
                    'race': race_ru,
                    'confidence': confidence,
                    'method': 'DeepFace'
                }
                
            except Exception as e:
                logger.debug(f"Ошибка при использовании DeepFace для определения расы: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при определении расы через DeepFace: {e}")
            return None
        finally:
            # Удаляем временный файл
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
        
        return None
    
    def _predict_race_insightface(self, image: np.ndarray, face_oval: np.ndarray) -> Optional[Dict]:
        """
        Определяет расу с использованием InsightFace
        
        Args:
            image: Изображение (BGR)
            face_oval: Массив точек овала лица
            
        Returns:
            Словарь с результатами или None
        """
        if not self.insightface_available or self.insightface_app is None:
            return None
        
        try:
            # InsightFace работает с RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Анализируем лицо
            faces = self.insightface_app.get(rgb_image)
            
            if not faces or len(faces) == 0:
                return None
            
            # Берем первое лицо с наибольшей уверенностью
            face = max(faces, key=lambda x: x.det_score if hasattr(x, 'det_score') else 1.0)
            
            # InsightFace может предоставлять атрибуты расы через age_gender
            # Но основная модель buffalo_l не всегда предоставляет race напрямую
            # Используем эмбеддинги для определения расы через кластеризацию
            
            # Если есть атрибут race, используем его
            if hasattr(face, 'race') and face.race is not None:
                race_mapping = {
                    'asian': 'Азиатская',
                    'indian': 'Индийская',
                    'black': 'Негроидная',
                    'white': 'Европеоидная',
                    'middle eastern': 'Ближневосточная',
                    'latino': 'Латиноамериканская',
                    'hispanic': 'Латиноамериканская'
                }
                
                race_str = str(face.race).lower()
                race_ru = None
                
                for key, value in race_mapping.items():
                    if key in race_str:
                        race_ru = value
                        break
                
                if race_ru:
                    confidence = float(face.det_score) if hasattr(face, 'det_score') else 0.75
                    return {
                        'race': race_ru,
                        'confidence': min(0.95, confidence),
                        'method': 'InsightFace'
                    }
            
            # Если race не доступен напрямую, возвращаем None
            # InsightFace в основном используется для распознавания лиц, а не для определения расы
            return None
            
        except Exception as e:
            logger.debug(f"Ошибка при использовании InsightFace для определения расы: {e}")
            return None
    
    def predict_race(self, image: np.ndarray, face_oval: np.ndarray) -> Dict[str, any]:
        """
        Определяет расу на изображении с использованием DeepFace и InsightFace
        
        Args:
            image: Исходное изображение (BGR)
            face_oval: Массив точек овала лица
            
        Returns:
            Словарь с результатами анализа расы
        """
        result = {
            'race': 'Не определена',
            'race_confidence': 0.0,
            'method': 'none'
        }
        
        # Пробуем DeepFace (приоритетный метод для расы)
        deepface_result = None
        if self.deepface_available:
            try:
                deepface_result = self._predict_race_deepface(image, face_oval)
                if deepface_result:
                    logger.info(f"DeepFace определил расу: {deepface_result['race']} "
                               f"(уверенность: {deepface_result['confidence']:.2f})")
            except Exception as e:
                logger.warning(f"Ошибка при использовании DeepFace для расы: {e}")
        
        # Пробуем InsightFace (резервный метод)
        insightface_result = None
        if self.insightface_available and not deepface_result:
            try:
                insightface_result = self._predict_race_insightface(image, face_oval)
                if insightface_result:
                    logger.info(f"InsightFace определил расу: {insightface_result['race']} "
                               f"(уверенность: {insightface_result['confidence']:.2f})")
            except Exception as e:
                logger.warning(f"Ошибка при использовании InsightFace для расы: {e}")
        
        # Выбираем лучший результат
        if deepface_result:
            result['race'] = deepface_result['race']
            result['race_confidence'] = deepface_result['confidence']
            result['method'] = 'DeepFace'
        elif insightface_result:
            result['race'] = insightface_result['race']
            result['race_confidence'] = insightface_result['confidence']
            result['method'] = 'InsightFace'
        
        return result
    
    def is_available(self) -> bool:
        """
        Проверяет, доступен ли анализатор расы
        
        Returns:
            True если хотя бы один метод доступен, False иначе
        """
        return self.deepface_available or self.insightface_available


