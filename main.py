"""
Analytical System - Face Comparison Module
Main script for running face comparison and recognition system

Repository: https://github.com/The-Open-Tech-Company/analytical-system
License: Unlicense (Open Source)
"""
import cv2
import sys
import os
from face_analyzer import FaceAnalyzer
from face_comparator import FaceComparator
from face_visualizer import FaceVisualizer


def main():
    """Основная функция для запуска системы сравнения лиц"""
    
    # Проверяем аргументы командной строки
    if len(sys.argv) < 3:
        print("Использование: python main.py <путь_к_изображению1> <путь_к_изображению2>")
        print("Пример: python main.py face1.jpg face2.jpg")
        sys.exit(1)
    
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    
    # Проверяем существование файлов
    if not os.path.exists(image1_path):
        print(f"Ошибка: Файл {image1_path} не найден!")
        sys.exit(1)
    
    if not os.path.exists(image2_path):
        print(f"Ошибка: Файл {image2_path} не найден!")
        sys.exit(1)
    
    print("Инициализация системы анализа лиц...")
    try:
        analyzer = FaceAnalyzer()
        comparator = FaceComparator()
        visualizer = FaceVisualizer()
    except Exception as e:
        print(f"Ошибка при инициализации системы: {e}")
        sys.exit(1)
    
    print(f"Извлечение характеристик из первого изображения: {image1_path}")
    try:
        features1 = analyzer.extract_face_features(image1_path)
    except Exception as e:
        print(f"Ошибка при обработке первого изображения: {e}")
        sys.exit(1)
    
    if features1 is None:
        print("Ошибка: Лицо не найдено на первом изображении!")
        print("Убедитесь, что на изображении четко видно одно лицо (анфас).")
        sys.exit(1)
    
    print(f"Извлечение характеристик из второго изображения: {image2_path}")
    try:
        features2 = analyzer.extract_face_features(image2_path)
    except Exception as e:
        print(f"Ошибка при обработке второго изображения: {e}")
        sys.exit(1)
    
    if features2 is None:
        print("Ошибка: Лицо не найдено на втором изображении!")
        print("Убедитесь, что на изображении четко видно одно лицо (анфас).")
        sys.exit(1)
    
    print("Сравнение лиц...")
    results = comparator.compare_faces(features1, features2)
    
    # Выводим результаты в консоль
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ ЛИЦ")
    print("="*50)
    
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
    
    for feature_name, similarity in results.items():
        if feature_name != 'overall':
            name_ru = feature_names_ru.get(feature_name, feature_name)
            print(f"{name_ru:30s}: {similarity:6.2f}%")
    
    print("-" * 50)
    overall = results.get('overall', 0.0)
    print(f"{feature_names_ru.get('overall', 'Общее совпадение'):30s}: {overall:6.2f}%")
    print("="*50)
    
    # Создаем визуализации
    print("\nСоздание визуализаций...")
    
    # Визуализация первого лица (зеленый)
    image1 = features1['image']
    vis1 = visualizer.visualize_face_features(image1, features1, visualizer.color_green)
    
    # Визуализация второго лица (красный)
    image2 = features2['image']
    vis2 = visualizer.visualize_face_features(image2, features2, visualizer.color_red)
    
    # Наложение двух рисунков
    overlay = visualizer.create_overlay_comparison(features1, features2, image1, image2, results)
    
    # Изображение с результатами
    results_img = visualizer.create_results_image(results)
    
    # Сохраняем результаты
    output_dir = "output"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Ошибка создания папки результатов: {e}")
        sys.exit(1)
    
    output1_path = os.path.join(output_dir, "face1_annotated.jpg")
    output2_path = os.path.join(output_dir, "face2_annotated.jpg")
    overlay_path = os.path.join(output_dir, "overlay_comparison.jpg")
    results_path = os.path.join(output_dir, "results.jpg")
    
    try:
        if not cv2.imwrite(output1_path, vis1):
            raise IOError(f"Не удалось сохранить {output1_path}")
        if not cv2.imwrite(output2_path, vis2):
            raise IOError(f"Не удалось сохранить {output2_path}")
        if not cv2.imwrite(overlay_path, overlay):
            raise IOError(f"Не удалось сохранить {overlay_path}")
        if not cv2.imwrite(results_path, results_img):
            raise IOError(f"Не удалось сохранить {results_path}")
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")
        sys.exit(1)
    
    print(f"\nРезультаты сохранены в папку '{output_dir}':")
    print(f"  - {output1_path} (первое лицо с зелеными линиями)")
    print(f"  - {output2_path} (второе лицо с красными линиями)")
    print(f"  - {overlay_path} (наложение двух рисунков)")
    print(f"  - {results_path} (текстовые результаты)")
    
    # Показываем изображения (опционально)
    print("\nОтображение результатов...")
    print("Нажмите любую клавишу для закрытия окон.")
    
    cv2.imshow("Первое лицо (зеленые линии)", vis1)
    cv2.imshow("Второе лицо (красные линии)", vis2)
    cv2.imshow("Наложение сравнения", overlay)
    cv2.imshow("Результаты сравнения", results_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nАнализ завершен!")


if __name__ == "__main__":
    main()

