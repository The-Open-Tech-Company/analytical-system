"""
Модуль для работы с базой данных лиц
"""
import sqlite3
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os


class FaceDatabase:
    """Класс для работы с базой данных лиц"""
    
    def __init__(self, db_path: str = "faces_database.db"):
        """
        Инициализирует базу данных
        
        Args:
            db_path: Путь к файлу базы данных
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Создает таблицу в базе данных, если она не существует"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Создаем таблицу для хранения лиц
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                birth_year INTEGER,
                additional_info TEXT,
                face_features BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_face(self, full_name: str, face_features: Dict, 
                 birth_year: Optional[int] = None, 
                 additional_info: Optional[str] = None) -> int:
        """
        Добавляет лицо в базу данных
        
        Args:
            full_name: ФИО человека
            face_features: Словарь с характеристиками лица (features)
            birth_year: Год рождения
            additional_info: Дополнительная информация
            
        Returns:
            ID добавленной записи
        """
        # Сериализуем features в pickle (для numpy массивов)
        features_blob = pickle.dumps(face_features)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO faces (full_name, birth_year, additional_info, face_features)
            VALUES (?, ?, ?, ?)
        ''', (full_name, birth_year, additional_info, features_blob))
        
        face_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return face_id
    
    def get_face(self, face_id: int) -> Optional[Dict]:
        """
        Получает лицо из базы данных по ID
        
        Args:
            face_id: ID лица
            
        Returns:
            Словарь с данными лица или None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, full_name, birth_year, additional_info, face_features, created_at
            FROM faces
            WHERE id = ?
        ''', (face_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        # Десериализуем features
        face_features = pickle.loads(row[4])
        
        return {
            'id': row[0],
            'full_name': row[1],
            'birth_year': row[2],
            'additional_info': row[3],
            'face_features': face_features,
            'created_at': row[5]
        }
    
    def get_all_faces(self) -> List[Dict]:
        """
        Получает все лица из базы данных
        
        Returns:
            Список словарей с данными лиц
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, full_name, birth_year, additional_info, face_features, created_at
            FROM faces
            ORDER BY created_at DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        faces = []
        for row in rows:
            # Десериализуем features
            face_features = pickle.loads(row[4])
            
            faces.append({
                'id': row[0],
                'full_name': row[1],
                'birth_year': row[2],
                'additional_info': row[3],
                'face_features': face_features,
                'created_at': row[5]
            })
        
        return faces
    
    def search_faces(self, query_features: Dict, threshold: float = 55.0) -> List[Tuple[Dict, float]]:
        """
        Ищет лица в базе данных, похожие на заданное
        
        Args:
            query_features: Характеристики лица для поиска
            threshold: Минимальный процент совпадения (по умолчанию 55%)
            
        Returns:
            Список кортежей (данные лица, процент совпадения), отсортированный по убыванию совпадения
        """
        from face_comparator import FaceComparator
        
        comparator = FaceComparator()
        all_faces = self.get_all_faces()
        
        results = []
        
        for face_data in all_faces:
            db_features = face_data['face_features']
            
            # Сравниваем лица
            comparison_result = comparator.compare_faces(query_features, db_features)
            overall_similarity = comparison_result.get('overall', 0.0)
            
            # Если совпадение >= порога, добавляем в результаты
            if overall_similarity >= threshold:
                results.append((face_data, overall_similarity))
        
        # Сортируем по убыванию совпадения
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def delete_face(self, face_id: int) -> bool:
        """
        Удаляет лицо из базы данных
        
        Args:
            face_id: ID лица
            
        Returns:
            True, если удаление успешно, False иначе
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM faces WHERE id = ?', (face_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return deleted
    
    def update_face(self, face_id: int, full_name: Optional[str] = None,
                   birth_year: Optional[int] = None,
                   additional_info: Optional[str] = None) -> bool:
        """
        Обновляет данные лица в базе данных
        
        Args:
            face_id: ID лица
            full_name: Новое ФИО (если указано)
            birth_year: Новый год рождения (если указан)
            additional_info: Новая дополнительная информация (если указана)
            
        Returns:
            True, если обновление успешно, False иначе
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = []
        values = []
        
        if full_name is not None:
            updates.append('full_name = ?')
            values.append(full_name)
        
        if birth_year is not None:
            updates.append('birth_year = ?')
            values.append(birth_year)
        
        if additional_info is not None:
            updates.append('additional_info = ?')
            values.append(additional_info)
        
        if not updates:
            conn.close()
            return False
        
        values.append(face_id)
        
        query = f'UPDATE faces SET {", ".join(updates)} WHERE id = ?'
        cursor.execute(query, values)
        
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return updated

