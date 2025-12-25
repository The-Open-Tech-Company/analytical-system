"""
Модуль для определения пола и возраста с использованием OpenCV DNN и предобученных моделей
"""
import os
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenderAgeDNN:
    """Класс для определения пола и возраста с использованием OpenCV DNN"""
    
    # URL для загрузки моделей (используем модели из репозитория OpenCV)
    # Примечание: Модели .caffemodel слишком большие для GitHub raw, поэтому используем альтернативные источники
    # или инструкции для ручной загрузки
    
    GENDER_PROTO_URLS = [
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy_gender.prototxt",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy_gender.prototxt",
    ]
    
    # Для .caffemodel файлов используем альтернативные источники или инструкции
    GENDER_MODEL_URLS = [
        # Попробуем найти рабочие зеркала или использовать инструкции для ручной загрузки
        "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/gender_net.caffemodel",
    ]
    
    AGE_PROTO_URLS = [
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy_age.prototxt",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy_age.prototxt",
    ]
    
    AGE_MODEL_URLS = [
        "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/age_net.caffemodel",
    ]
    
    # Инструкции для ручной загрузки (если автоматическая не работает)
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Если автоматическая загрузка не работает, загрузите модели вручную:
    
    1. Перейдите на https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
    2. Скачайте следующие файлы:
       - deploy_gender.prototxt
       - deploy_age.prototxt
    3. Перейдите на https://github.com/opencv/opencv_extra/tree/master/testdata/dnn
    4. Скачайте следующие файлы:
       - gender_net.caffemodel
       - age_net.caffemodel
    5. Поместите все файлы в папку 'models/' в корне проекта
    
    Альтернативно, используйте Git LFS для клонирования репозитория OpenCV.
    """
    
    def __init__(self, models_dir: str = "models", swap_gender_classes: bool = True):
        """
        Инициализация класса
        
        Args:
            models_dir: Директория для хранения моделей
            swap_gender_classes: Если True, меняет местами классы пола (для моделей с обратным порядком)
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Пути к файлам моделей
        self.gender_proto = os.path.join(models_dir, "deploy_gender.prototxt")
        self.gender_model = os.path.join(models_dir, "gender_net.caffemodel")
        self.age_proto = os.path.join(models_dir, "deploy_age.prototxt")
        self.age_model = os.path.join(models_dir, "age_net.caffemodel")
        
        # Загружаем модели
        self.gender_net = None
        self.age_net = None
        
        # Флаг для переключения порядка классов пола
        # Некоторые модели имеют порядок [Female, Male] вместо [Male, Female]
        self.swap_gender_classes = swap_gender_classes
        
        # Список возрастных категорий (модель Adience)
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', 
                        '(38-43)', '(48-53)', '(60-100)']
        
        # Средние значения для нормализации (BGR)
        self.model_mean = (78.4263377603, 87.7689143744, 114.895847746)
        
        self._load_models()
    
    def _download_file(self, urls: list, filepath: str) -> bool:
        """
        Скачивает файл по URL (пробует несколько URL)
        
        Args:
            urls: Список URL для попытки загрузки
            filepath: Путь для сохранения
            
        Returns:
            True если успешно, False иначе
        """
        if isinstance(urls, str):
            urls = [urls]
        
        for url in urls:
            try:
                logger.info(f"Загрузка {os.path.basename(filepath)} с {url}...")
                urllib.request.urlretrieve(url, filepath)
                
                # Проверяем, что файл не пустой
                if os.path.getsize(filepath) > 0:
                    logger.info(f"Файл {os.path.basename(filepath)} успешно загружен")
                    return True
                else:
                    logger.warning(f"Загруженный файл пуст: {url}")
                    os.remove(filepath)
            except Exception as e:
                logger.warning(f"Ошибка при загрузке {url}: {e}")
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                continue
        
        return False
    
    def _load_models(self):
        """Загружает модели для определения пола и возраста"""
        # Проверяем наличие файлов моделей
        models_loaded = True
        
        # Загружаем модель для определения пола
        if not os.path.exists(self.gender_proto) or not os.path.exists(self.gender_model):
            logger.info("Модели для определения пола не найдены. Попытка загрузки...")
            if not os.path.exists(self.gender_proto):
                if not self._download_file(self.GENDER_PROTO_URLS, self.gender_proto):
                    logger.error("Не удалось загрузить файл конфигурации модели пола")
                    logger.info("Попробуйте загрузить вручную. См. инструкции в коде.")
                    models_loaded = False
            
            if not os.path.exists(self.gender_model):
                if not self._download_file(self.GENDER_MODEL_URLS, self.gender_model):
                    logger.error("Не удалось загрузить модель для определения пола")
                    logger.warning(self.MANUAL_DOWNLOAD_INSTRUCTIONS)
                    models_loaded = False
        
        # Загружаем модель для определения возраста
        if not os.path.exists(self.age_proto) or not os.path.exists(self.age_model):
            logger.info("Модели для определения возраста не найдены. Попытка загрузки...")
            if not os.path.exists(self.age_proto):
                if not self._download_file(self.AGE_PROTO_URLS, self.age_proto):
                    logger.error("Не удалось загрузить файл конфигурации модели возраста")
                    logger.info("Попробуйте загрузить вручную. См. инструкции в коде.")
                    models_loaded = False
            
            if not os.path.exists(self.age_model):
                if not self._download_file(self.AGE_MODEL_URLS, self.age_model):
                    logger.error("Не удалось загрузить модель для определения возраста")
                    logger.warning(self.MANUAL_DOWNLOAD_INSTRUCTIONS)
                    models_loaded = False
        
        # Загружаем модели в OpenCV DNN
        if models_loaded:
            try:
                if os.path.exists(self.gender_proto) and os.path.exists(self.gender_model):
                    self.gender_net = cv2.dnn.readNetFromCaffe(self.gender_proto, self.gender_model)
                    logger.info("Модель для определения пола загружена успешно")
                else:
                    logger.warning("Файлы модели пола не найдены")
            except Exception as e:
                logger.error(f"Ошибка при загрузке модели пола: {e}")
                self.gender_net = None
            
            try:
                if os.path.exists(self.age_proto) and os.path.exists(self.age_model):
                    self.age_net = cv2.dnn.readNetFromCaffe(self.age_proto, self.age_model)
                    logger.info("Модель для определения возраста загружена успешно")
                else:
                    logger.warning("Файлы модели возраста не найдены")
            except Exception as e:
                logger.error(f"Ошибка при загрузке модели возраста: {e}")
                self.age_net = None
        else:
            logger.warning("Некоторые модели не были загружены. Функционал будет ограничен.")
    
    def _extract_face_roi(self, image: np.ndarray, face_oval: np.ndarray) -> Optional[np.ndarray]:
        """
        Извлекает область лица из изображения
        
        Args:
            image: Исходное изображение
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
        
        # Добавляем отступы (20% с каждой стороны)
        padding_x = int((x_max - x_min) * 0.2)
        padding_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(image.shape[1], x_max + padding_x)
        y_max = min(image.shape[0], y_max + padding_y)
        
        # Извлекаем область лица
        face_roi = image[y_min:y_max, x_min:x_max]
        
        if face_roi.size == 0:
            return None
        
        return face_roi
    
    def predict_gender_age(self, image: np.ndarray, face_oval: np.ndarray) -> Dict[str, any]:
        """
        Определяет пол и возраст на изображении
        
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
        
        # Извлекаем область лица
        face_roi = self._extract_face_roi(image, face_oval)
        if face_roi is None:
            logger.warning("Не удалось извлечь область лица")
            return result
        
        # Проверяем размер области лица
        if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
            logger.warning(f"Область лица слишком мала: {face_roi.shape}")
            return result
        
        # Подготавливаем изображение для модели (227x227 для моделей Adience)
        try:
            # Убеждаемся, что изображение в правильном формате (BGR, uint8)
            if face_roi.dtype != np.uint8:
                face_roi = face_roi.astype(np.uint8)
            
            # Проверяем, что изображение не пустое
            if face_roi.size == 0:
                logger.error("Область лица пуста")
                return result
            
            # Создаем blob для модели
            # Модель Adience использует mean subtraction и размер 227x227
            blob = cv2.dnn.blobFromImage(
                face_roi, 
                scalefactor=1.0,  # Не масштабируем значения (они уже в [0, 255])
                size=(227, 227),
                mean=self.model_mean,  # Вычитаем средние значения (BGR)
                swapRB=False,  # Не меняем местами R и B (изображение уже в BGR)
                crop=False  # Не обрезаем, а масштабируем с сохранением пропорций
            )
            
            # Логируем информацию о blob для отладки
            logger.debug(f"Blob shape: {blob.shape}, min: {blob.min():.2f}, max: {blob.max():.2f}, mean: {blob.mean():.2f}")
            
        except Exception as e:
            logger.error(f"Ошибка при подготовке изображения: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return result
        
        # Определяем пол
        if self.gender_net is not None:
            try:
                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                
                # Проверяем форму выходных данных
                preds_flat = gender_preds.flatten()
                
                # Логируем сырые значения для отладки
                logger.info(f"Сырые значения модели пола: {preds_flat}")
                
                # Проверяем, являются ли значения уже вероятностями (сумма близка к 1) или logits
                sum_preds = np.sum(preds_flat)
                is_probability = abs(sum_preds - 1.0) < 0.1
                
                if is_probability:
                    # Уже вероятности, используем как есть
                    normalized_preds = preds_flat
                else:
                    # Это logits, применяем softmax
                    exp_preds = np.exp(preds_flat - np.max(preds_flat))
                    normalized_preds = exp_preds / np.sum(exp_preds)
                
                # Убеждаемся, что у нас есть 2 класса
                if len(normalized_preds) < 2:
                    logger.warning(f"Модель вернула неожиданное количество классов: {len(normalized_preds)}")
                    # Если только один класс, создаем второй с нулевой вероятностью
                    if len(normalized_preds) == 1:
                        normalized_preds = np.array([normalized_preds[0], 1.0 - normalized_preds[0]])
                
                # Получаем вероятности для обоих классов
                # Если swap_gender_classes=True, меняем местами интерпретацию классов
                if self.swap_gender_classes:
                    # Класс 0 = Female, класс 1 = Male
                    female_prob = float(normalized_preds[0])
                    male_prob = float(normalized_preds[1]) if len(normalized_preds) > 1 else 1.0 - female_prob
                else:
                    # Класс 0 = Male, класс 1 = Female (стандартный порядок для Adience)
                    male_prob = float(normalized_preds[0])
                    female_prob = float(normalized_preds[1]) if len(normalized_preds) > 1 else 1.0 - male_prob
                
                # Проверяем, что вероятности различаются (модель не всегда возвращает одинаковые значения)
                prob_difference = abs(male_prob - female_prob)
                
                # Определяем пол на основе максимальной вероятности
                # Если разница очень мала (< 0.05), пол не может быть определен точно
                if prob_difference < 0.05:
                    # Если вероятности почти равны, пол не определен
                    result['gender'] = 'Не определен'
                    gender_confidence = 0.0
                    logger.warning(f"DNN вернул почти одинаковые вероятности: Male={male_prob:.3f}, Female={female_prob:.3f}. "
                                 f"Разница слишком мала ({prob_difference:.3f}), пол не может быть определен.")
                elif male_prob > female_prob:
                    result['gender'] = 'Мужской'
                    # Если разница мала, но все же есть, снижаем уверенность
                    if prob_difference < 0.1:
                        gender_confidence = 0.5 + prob_difference * 0.5
                    else:
                        gender_confidence = max(male_prob, female_prob)
                else:
                    result['gender'] = 'Женский'
                    # Если разница мала, но все же есть, снижаем уверенность
                    if prob_difference < 0.1:
                        gender_confidence = 0.5 + prob_difference * 0.5
                    else:
                        gender_confidence = max(male_prob, female_prob)
                
                # Убеждаемся, что уверенность в диапазоне [0, 1]
                gender_confidence = max(0.0, min(1.0, gender_confidence))
                
                result['gender_confidence'] = gender_confidence
                
                # Логируем для отладки
                logger.info(f"DNN определил пол: {result['gender']}, уверенность: {gender_confidence:.2f}, "
                           f"вероятности: Male={male_prob:.3f}, Female={female_prob:.3f}, разница={prob_difference:.3f}")
            except Exception as e:
                logger.error(f"Ошибка при определении пола: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Определяем возраст
        if self.age_net is not None:
            try:
                self.age_net.setInput(blob)
                age_preds = self.age_net.forward()
                
                # Проверяем форму выходных данных
                preds_flat = age_preds.flatten()
                
                # Логируем сырые значения для отладки
                logger.info(f"Сырые значения модели возраста: {preds_flat[:5]}... (показаны первые 5)")
                
                # Проверяем, являются ли значения уже вероятностями (сумма близка к 1) или logits
                sum_preds = np.sum(preds_flat)
                is_probability = abs(sum_preds - 1.0) < 0.1
                
                if is_probability:
                    # Уже вероятности, используем как есть
                    normalized_preds = preds_flat
                else:
                    # Это logits, применяем softmax
                    exp_preds = np.exp(preds_flat - np.max(preds_flat))
                    normalized_preds = exp_preds / np.sum(exp_preds)
                
                # Получаем индекс класса с максимальной вероятностью
                age_idx = normalized_preds.argmax()
                age_confidence = float(normalized_preds[age_idx])
                
                # Убеждаемся, что уверенность в диапазоне [0, 1]
                age_confidence = max(0.0, min(1.0, age_confidence))
                
                # Получаем возрастную категорию
                if age_idx < len(self.age_list):
                    age_category = self.age_list[age_idx]
                    result['age_category'] = age_category
                    
                    # Преобразуем категорию в числовой возраст (среднее значение диапазона)
                    age_ranges = {
                        '(0-2)': 1,
                        '(4-6)': 5,
                        '(8-12)': 10,
                        '(15-20)': 17,
                        '(25-32)': 28,
                        '(38-43)': 40,
                        '(48-53)': 50,
                        '(60-100)': 70
                    }
                    
                    estimated_age = age_ranges.get(age_category, 30)
                    result['age'] = f"{estimated_age} ± 5 лет"
                    result['age_confidence'] = age_confidence
                    
                    logger.info(f"DNN определил возраст: {age_category} ({estimated_age} лет), уверенность: {age_confidence:.2f}")
                else:
                    logger.warning(f"Индекс возраста {age_idx} выходит за пределы списка категорий (длина: {len(self.age_list)})")
            except Exception as e:
                logger.error(f"Ошибка при определении возраста: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        return result
    
    def is_available(self) -> bool:
        """
        Проверяет, доступны ли модели
        
        Returns:
            True если модели загружены, False иначе
        """
        return self.gender_net is not None and self.age_net is not None

