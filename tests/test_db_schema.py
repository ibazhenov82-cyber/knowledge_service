"""Проверка схемы файла базы при старте: старая база — понятная ошибка, а не «no such column» при загрузке."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from knowledge_service.db import Database, SchemaMismatchError


class SchemaCheckTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "knowledge.sqlite")

    def tearDown(self):
        self.dir.cleanup()

    def test_fresh_and_reopened_database(self):
        Database(self.path)
        Database(self.path)  # повторное открытие той же схемы — без ошибок

    def test_old_database_is_reported(self):
        Database(self.path)
        conn = sqlite3.connect(self.path)
        conn.execute("ALTER TABLE documents DROP COLUMN properties_json")
        conn.commit()
        conn.close()
        with self.assertRaises(SchemaMismatchError) as ctx:
            Database(self.path)
        message = str(ctx.exception)
        self.assertIn("documents: нет колонок properties_json", message)
        self.assertIn("удалите файл базы", message)


if __name__ == "__main__":
    unittest.main()
