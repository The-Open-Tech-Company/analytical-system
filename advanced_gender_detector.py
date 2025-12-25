"""
Продвинутый модуль для определения пола с использованием множества специализированных библиотек
Использует ансамблевый подход для максимальной точности
"""
import os
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импорт DeepFace
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("DeepFace успешно импортирован")
except ImportError:
    logger.warning("DeepFace не доступен")
except Exception as e:
    logger.warning(f"Ошибка при импорте DeepFace: {e}")

# Импорт RetinaFace для более точной детекции
RETINAFACE_AVAILABLE = False
try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
    logger.info("RetinaFace успешно импортирован")
except ImportError:
    logger.warning("RetinaFace не доступен. Установите: pip install retina-face")
except Exception as e:
    logger.warning(f"Ошибка при импорте RetinaFace: {e}")

# Импорт MTCNN
MTCNN_AVAILABLE = False
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
    logger.info("MTCNN успешно импортирован")
except ImportError:
    logger.warning("MTCNN не доступен")
except Exception as e:
    logger.warning(f"Ошибка при импорте MTCNN: {e}")


class AdvancedGenderDetector:
    """Продвинутый детектор пола с использованием множества специализированных библиотек"""
    
    def __init__(self):
        """Инициализация детектора"""
        self.deepface_models = []
        self.deepface_backends = []
        
        # Инициализируем доступные модели DeepFace
        if DEEPFACE_AVAILABLE:
            # Используем несколько самых точных моделей (проверяем доступность при первом использовании)
            # Приоритет: ArcFace (самый точный), затем VGG-Face, Facenet, DeepFace
            self.deepface_models = ['ArcFace', 'VGG-Face', 'Facenet', 'DeepFace']
            logger.info(f"Доступные модели DeepFace для проверки: {self.deepface_models}")
            
            # Определяем доступные бэкенды для детекции (приоритет: retinaface, opencv, ssd, dlib, mtcnn)
            # RetinaFace обычно самый точный для детекции лиц
            self.deepface_backends = []
            # Проверяем доступность бэкендов
            backend_priority = ['retinaface', 'opencv', 'ssd', 'dlib', 'mtcnn']
            for backend in backend_priority:
                # Просто добавляем в список - проверка будет при использовании
                self.deepface_backends.append(backend)
            logger.info(f"Бэкенды DeepFace для использования: {self.deepface_backends}")
        
        # RetinaFace не требует предварительной инициализации модели
        # Модель загружается автоматически при первом вызове detect_faces
        if RETINAFACE_AVAILABLE:
            logger.info("RetinaFace доступен (модель загрузится автоматически)")
        
        # Инициализируем MTCNN
        self.mtcnn_detector = None
        if MTCNN_AVAILABLE:
            try:
                self.mtcnn_detector = MTCNN()
                logger.info("MTCNN инициализирован")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать MTCNN: {e}")
        
        logger.info(f"AdvancedGenderDetector инициализирован. "
                   f"Модели DeepFace: {len(self.deepface_models)}, "
                   f"Бэкенды: {len(self.deepface_backends)}")
    
    def _extract_face_with_retinaface(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Извлекает лицо с использованием RetinaFace"""
        if not RETINAFACE_AVAILABLE:
            return None
        
        try:
            # RetinaFace работает с RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Детектируем лицо
            faces = RetinaFace.detect_faces(rgb_image)
            
            if faces:
                # Берем первое лицо
                face_key = list(faces.keys())[0]
                face_data = faces[face_key]
                facial_area = face_data['facial_area']
                
                x1, y1, x2, y2 = facial_area
                # Добавляем отступы
                padding = int((x2 - x1) * 0.2)
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                face_roi = image[y1:y2, x1:x2]
                if face_roi.size > 0:
                    return face_roi
        except Exception as e:
            logger.debug(f"Ошибка при использовании RetinaFace: {e}")
        
        return None
    
    def _extract_face_with_mtcnn(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Извлекает лицо с использованием MTCNN"""
        if not MTCNN_AVAILABLE or self.mtcnn_detector is None:
            return None
        
        try:
            # MTCNN работает с RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Детектируем лицо
            faces = self.mtcnn_detector.detect_faces(rgb_image)
            
            if faces:
                # Берем первое лицо с наибольшей уверенностью
                face = max(faces, key=lambda x: x['confidence'])
                x, y, w, h = face['box']
                
                # Добавляем отступы
                padding = int(w * 0.2)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)
                
                face_roi = image[y1:y2, x1:x2]
                if face_roi.size > 0:
                    return face_roi
        except Exception as e:
            logger.debug(f"Ошибка при использовании MTCNN: {e}")
        
        return None
    
    def _predict_with_deepface(self, image_path: str, model_name: str = None, 
                               detector_backend: str = None) -> Optional[Dict]:
        """Предсказывает пол с использованием DeepFace"""
        if not DEEPFACE_AVAILABLE:
            return None
        
        try:
            # Используем лучший доступный бэкенд или указанный
            backend = detector_backend or (self.deepface_backends[0] if self.deepface_backends else 'opencv')
            model = model_name or (self.deepface_models[0] if self.deepface_models else 'VGG-Face')
            
            analysis = DeepFace.analyze(
                img_path=image_path,
                actions=['gender'],
                model_name=model,
                detector_backend=backend,
                enforce_detection=False,
                silent=True
            )
            
            # Обрабатываем результат
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            gender_result = analysis.get('dominant_gender', '') or analysis.get('gender', '')
            gender_probs = analysis.get('gender', {})
            
            # Если gender_result - словарь, извлекаем ключ с максимальной вероятностью
            if isinstance(gender_result, dict):
                if gender_result:
                    gender_result = max(gender_result, key=gender_result.get)
                    gender_probs = gender_result
                else:
                    return None
            
            # Определяем пол
            gender_str = str(gender_result).lower()
            if 'man' in gender_str or 'male' in gender_str:
                gender = 'Мужской'
                if isinstance(gender_probs, dict):
                    confidence = float(gender_probs.get('Man', gender_probs.get('Male', 
                                    gender_probs.get('man', gender_probs.get('male', 0.5)))))
                else:
                    confidence = 0.85
            elif 'woman' in gender_str or 'female' in gender_str:
                gender = 'Женский'
                if isinstance(gender_probs, dict):
                    confidence = float(gender_probs.get('Woman', gender_probs.get('Female',
                                    gender_probs.get('woman', gender_probs.get('female', 0.5)))))
                else:
                    confidence = 0.85
            else:
                return None
            
            return {
                'gender': gender,
                'confidence': confidence,
                'model': model,
                'backend': backend
            }
        except Exception as e:
            logger.debug(f"Ошибка при использовании DeepFace ({model_name}, {detector_backend}): {e}")
            return None
    
    def predict_gender_ensemble(self, image: np.ndarray, face_oval: np.ndarray = None) -> Dict[str, any]:
        """
        Определяет пол используя ансамбль методов для максимальной точности
        
        Args:
            image: Изображение (BGR)
            face_oval: Массив точек овала лица (опционально)
            
        Returns:
            Словарь с результатами
        """
        result = {
            'gender': 'Не определен',
            'confidence': 0.0,
            'method': 'ensemble',
            'votes': {'Мужской': 0, 'Женский': 0},
            'details': []
        }
        
        # Собираем все предсказания
        predictions = []
        
        # Извлекаем область лица разными методами
        face_rois = []
        
        # Метод 1: Используем переданный face_oval
        if face_oval is not None and len(face_oval) > 0:
            x_min = int(np.min(face_oval[:, 0]))
            y_min = int(np.min(face_oval[:, 1]))
            x_max = int(np.max(face_oval[:, 0]))
            y_max = int(np.max(face_oval[:, 1]))
            padding_x = int((x_max - x_min) * 0.3)
            padding_y = int((y_max - y_min) * 0.3)
            x_min = max(0, x_min - padding_x)
            y_min = max(0, y_min - padding_y)
            x_max = min(image.shape[1], x_max + padding_x)
            y_max = min(image.shape[0], y_max + padding_y)
            face_roi = image[y_min:y_max, x_min:x_max]
            if face_roi.size > 0:
                face_rois.append(('face_oval', face_roi))
        
        # Метод 2: RetinaFace
        retinaface_roi = self._extract_face_with_retinaface(image)
        if retinaface_roi is not None:
            face_rois.append(('retinaface', retinaface_roi))
        
        # Метод 3: MTCNN
        mtcnn_roi = self._extract_face_with_mtcnn(image)
        if mtcnn_roi is not None:
            face_rois.append(('mtcnn', mtcnn_roi))
        
        # Если нет извлеченных лиц, используем все изображение
        if not face_rois:
            face_rois.append(('full_image', image))
        
        # Для каждого извлеченного лица используем все доступные модели DeepFace
        temp_files = []
        try:
            for roi_name, face_roi in face_rois:
                # Сохраняем во временный файл
                temp_path = None
                try:
                    import tempfile as tf
                    from PIL import Image
                    temp_dir = tf.gettempdir()
                    temp_path = os.path.join(temp_dir, f"gender_det_{roi_name}_{os.getpid()}.jpg")
                    
                    rgb_image = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    pil_image.save(temp_path, "JPEG", quality=95)
                    temp_files.append(temp_path)
                    
                    # Используем все доступные модели DeepFace
                    for model in self.deepface_models:
                        # Пробуем разные бэкенды для каждой модели (используем 2-3 лучших)
                        backends_to_try = self.deepface_backends[:3] if len(self.deepface_backends) >= 3 else self.deepface_backends
                        for backend in backends_to_try:
                            try:
                                pred = self._predict_with_deepface(temp_path, model, backend)
                                if pred:
                                    pred['roi_source'] = roi_name
                                    predictions.append(pred)
                                    result['details'].append(f"{model}+{backend} ({roi_name}): {pred['gender']} ({pred['confidence']:.2f})")
                            except Exception as e:
                                logger.debug(f"Ошибка при использовании {model}+{backend}: {e}")
                                continue
                    
                    # Также пробуем без указания модели (автовыбор) с лучшим бэкендом
                    if self.deepface_backends:
                        try:
                            pred = self._predict_with_deepface(temp_path, None, self.deepface_backends[0])
                            if pred:
                                pred['roi_source'] = roi_name
                                predictions.append(pred)
                                result['details'].append(f"DeepFace-auto ({roi_name}): {pred['gender']} ({pred['confidence']:.2f})")
                        except Exception as e:
                            logger.debug(f"Ошибка при использовании DeepFace-auto: {e}")
                
                except Exception as e:
                    logger.debug(f"Ошибка при обработке {roi_name}: {e}")
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
        
        finally:
            # Удаляем временные файлы
            for temp_path in temp_files:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
        
        # Если нет предсказаний, возвращаем результат по умолчанию
        if not predictions:
            logger.warning("Не удалось получить предсказания от специализированных библиотек")
            return result
        
        # Ансамблевое голосование с учетом уверенности
        male_votes = 0.0
        female_votes = 0.0
        total_confidence = 0.0
        
        for pred in predictions:
            confidence = pred['confidence']
            gender = pred['gender']
            
            if gender == 'Мужской':
                male_votes += confidence
            elif gender == 'Женский':
                female_votes += confidence
            
            total_confidence += confidence
        
        # Подсчитываем количество голосов для каждого пола
        male_count = len([p for p in predictions if p['gender'] == 'Мужской'])
        female_count = len([p for p in predictions if p['gender'] == 'Женский'])
        result['votes']['Мужской'] = male_count
        result['votes']['Женский'] = female_count
        
        # Вычисляем разницу между голосами
        votes_difference = abs(male_votes - female_votes)
        total_votes = male_votes + female_votes
        
        # Определяем итоговый пол
        # Если разница очень мала (< 5% от общего количества голосов), пол не может быть определен точно
        if total_votes > 0 and votes_difference / total_votes < 0.05:
            # Если голоса почти равны, пол не определен
            result['gender'] = 'Не определен'
            result['confidence'] = 0.0
            logger.warning(f"Ансамбль: голоса почти равны (Male={male_votes:.2f}, Female={female_votes:.2f}), "
                         f"пол не может быть определен точно")
        elif male_votes > female_votes:
            result['gender'] = 'Мужской'
            result['confidence'] = min(0.98, male_votes / total_votes)
        elif female_votes > male_votes:
            result['gender'] = 'Женский'
            result['confidence'] = min(0.98, female_votes / total_votes)
        else:
            # Если голоса абсолютно равны, пол не определен
            result['gender'] = 'Не определен'
            result['confidence'] = 0.0
            logger.warning("Ансамбль: голоса абсолютно равны, пол не может быть определен")
        
        logger.info(f"Ансамбль определил пол: {result['gender']} "
                   f"(уверенность: {result['confidence']:.2f}, "
                   f"голосов: {result['votes']}, "
                   f"всего предсказаний: {len(predictions)})")
        
        return result
    
    def is_available(self) -> bool:
        """Проверяет доступность детектора"""
        return (DEEPFACE_AVAILABLE and len(self.deepface_models) > 0) or \
               RETINAFACE_AVAILABLE or \
               MTCNN_AVAILABLE

