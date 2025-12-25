"""
Модуль для определения пола и возраста с использованием библиотеки DeepFace
DeepFace использует современные модели глубокого обучения для более точного определения
"""
import os
import cv2
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Пытаемся импортировать deepface
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("DeepFace успешно импортирован")
except ImportError as e:
    logger.warning(f"DeepFace не доступен: {e}. Установите: pip install deepface tensorflow")
    DEEPFACE_AVAILABLE = False
except Exception as e:
    logger.warning(f"Ошибка при импорте DeepFace: {e}")
    DEEPFACE_AVAILABLE = False


class DeepFaceGenderAnalyzer:
    """Класс для определения пола и возраста с использованием DeepFace"""
    
    def __init__(self, model_name: str = "VGG-Face"):
        """
        Инициализация класса
        
        Args:
            model_name: Название модели DeepFace для использования
                       Доступные модели: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace
                       VGG-Face - хороший баланс точности и скорости
                       ArcFace - самая точная, но медленнее
        """
        self.model_name = model_name
        self._is_available_flag = DEEPFACE_AVAILABLE
        
        if not self._is_available_flag:
            logger.warning("DeepFace недоступен. Установите: pip install deepface tensorflow")
        else:
            logger.info(f"DeepFaceGenderAnalyzer инициализирован с моделью: {model_name}")
    
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
        
        # Добавляем отступы (30% с каждой стороны для лучшего качества)
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
        Сохраняет изображение во временный файл для DeepFace
        
        Args:
            image: Изображение (BGR)
            
        Returns:
            Путь к временному файлу или None
        """
        try:
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"deepface_temp_{os.getpid()}.jpg")
            
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
    
    def predict_gender_age(self, image: np.ndarray, face_oval: np.ndarray) -> Dict[str, any]:
        """
        Определяет пол и возраст на изображении с использованием DeepFace
        
        Args:
            image: Исходное изображение (BGR)
            face_oval: Массив точек овала лица
            
        Returns:
            Словарь с результатами анализа
        """
        result = {
            'gender': 'Не определен',
            'age': 'Не определен',
            'gender_confidence': 0.0,
            'age_confidence': 0.0,
            'age_category': None
        }
        
        if not self._is_available_flag:
            logger.warning("DeepFace недоступен")
            return result
        
        # Извлекаем область лица
        face_roi = self._extract_face_roi(image, face_oval)
        if face_roi is None:
            logger.warning("Не удалось извлечь область лица")
            return result
        
        # Проверяем размер области лица
        if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
            logger.warning(f"Область лица слишком мала: {face_roi.shape}")
            return result
        
        temp_path = None
        try:
            # Сохраняем изображение во временный файл
            temp_path = self._save_temp_image(face_roi)
            if temp_path is None:
                logger.error("Не удалось сохранить временное изображение")
                return result
            
            # Используем DeepFace для анализа
            # actions=['gender', 'age'] - определяем пол и возраст
            # enforce_detection=False - не требуем обязательного обнаружения лица (уже извлекли)
            # silent=True - не выводим предупреждения
            try:
                # Для DeepFace 0.0.95 используем обновленный API
                # Пробуем использовать detector_backend='skip' если поддерживается
                # Если нет, используем enforce_detection=False
                try:
                    # В новых версиях DeepFace может потребоваться другой формат
                    analysis = DeepFace.analyze(
                        img_path=temp_path,
                        actions=['gender', 'age'],
                        model_name=self.model_name,
                        enforce_detection=False,
                        silent=True,
                        detector_backend='skip'  # Пропускаем детекцию, так как уже извлекли лицо
                    )
                except (TypeError, ValueError, KeyError) as e:
                    # Если detector_backend не поддерживается, используем только enforce_detection
                    logger.debug(f"detector_backend='skip' не поддерживается, используем enforce_detection=False: {e}")
                    try:
                        analysis = DeepFace.analyze(
                            img_path=temp_path,
                            actions=['gender', 'age'],
                            model_name=self.model_name,
                            enforce_detection=False,
                            silent=True
                        )
                    except Exception as e2:
                        # Если и это не работает, пробуем без указания model_name
                        logger.debug(f"Пробуем без указания model_name: {e2}")
                        analysis = DeepFace.analyze(
                            img_path=temp_path,
                            actions=['gender', 'age'],
                            enforce_detection=False,
                            silent=True
                        )
                
                # DeepFace возвращает список или словарь в зависимости от версии
                if isinstance(analysis, list):
                    analysis = analysis[0]
                elif not isinstance(analysis, dict):
                    # Если это не словарь и не список, пробуем преобразовать
                    logger.warning(f"Неожиданный тип результата от DeepFace: {type(analysis)}")
                    return result
                
                # Извлекаем результаты (поддержка разных форматов ответа DeepFace)
                gender_result = analysis.get('dominant_gender', '') or analysis.get('gender', '')
                age_result = analysis.get('age', None)
                
                # Если gender_result - это словарь с вероятностями, извлекаем dominant_gender
                if isinstance(gender_result, dict):
                    # Находим ключ с максимальной вероятностью
                    if gender_result:
                        dominant_key = max(gender_result, key=gender_result.get)
                        gender_result = dominant_key
                        gender_probs = gender_result
                    else:
                        gender_result = ''
                        gender_probs = {}
                else:
                    # Получаем вероятности отдельно
                    gender_probs = analysis.get('gender', {})
                
                # Преобразуем пол в нужный формат
                if gender_result:
                    gender_str = str(gender_result).lower()
                    if 'man' in gender_str or 'male' in gender_str:
                        result['gender'] = 'Мужской'
                        # Получаем уверенность из вероятностей
                        if isinstance(gender_probs, dict):
                            male_prob = gender_probs.get('Man', gender_probs.get('Male', gender_probs.get('man', gender_probs.get('male', 0.5))))
                            result['gender_confidence'] = float(male_prob)
                        else:
                            result['gender_confidence'] = 0.85  # По умолчанию высокая уверенность
                    elif 'woman' in gender_str or 'female' in gender_str:
                        result['gender'] = 'Женский'
                        # Получаем уверенность из вероятностей
                        if isinstance(gender_probs, dict):
                            female_prob = gender_probs.get('Woman', gender_probs.get('Female', gender_probs.get('woman', gender_probs.get('female', 0.5))))
                            result['gender_confidence'] = float(female_prob)
                        else:
                            result['gender_confidence'] = 0.85  # По умолчанию высокая уверенность
                    else:
                        logger.warning(f"Неожиданный результат пола от DeepFace: {gender_result}")
                
                # Преобразуем возраст
                if age_result is not None:
                    try:
                        age = int(float(age_result))
                        result['age'] = f"{age} ± 5 лет"
                        result['age_confidence'] = 0.80  # DeepFace обычно дает хорошую точность для возраста
                        
                        # Определяем возрастную категорию
                        if age < 3:
                            result['age_category'] = '(0-2)'
                        elif age < 7:
                            result['age_category'] = '(4-6)'
                        elif age < 14:
                            result['age_category'] = '(8-12)'
                        elif age < 22:
                            result['age_category'] = '(15-20)'
                        elif age < 35:
                            result['age_category'] = '(25-32)'
                        elif age < 45:
                            result['age_category'] = '(38-43)'
                        elif age < 56:
                            result['age_category'] = '(48-53)'
                        else:
                            result['age_category'] = '(60-100)'
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Ошибка при преобразовании возраста: {e}")
                
                logger.info(f"DeepFace определил - Пол: {result['gender']} "
                           f"(уверенность: {result['gender_confidence']:.2f}), "
                           f"Возраст: {result['age']} (уверенность: {result['age_confidence']:.2f})")
                
            except Exception as e:
                logger.error(f"Ошибка при анализе DeepFace: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
        except Exception as e:
            logger.error(f"Ошибка при определении пола/возраста через DeepFace: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Удаляем временный файл
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл {temp_path}: {e}")
        
        return result
    
    def is_available(self) -> bool:
        """
        Проверяет, доступен ли DeepFace
        
        Returns:
            True если DeepFace доступен, False иначе
        """
        return self._is_available_flag

