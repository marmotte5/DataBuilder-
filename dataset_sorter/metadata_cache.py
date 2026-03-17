"""SQLite-backed metadata cache for fast dataset queries.

Stores image metadata (file path, size, mtime, hash, tags, token count,
bucket assignment) in a local SQLite database. Provides O(1) lookups
and rich queries like "show all images with >77 tokens" instantly.

Replaces per-file JSON sidecars with a single DB per dataset.
"""

import hashlib
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    path        TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    size_bytes  INTEGER,
    mtime       REAL,
    width       INTEGER,
    height      INTEGER,
    md5         TEXT,
    ahash       TEXT,
    tags        TEXT,
    token_count INTEGER,
    bucket      INTEGER,
    cached_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_md5 ON metadata(md5);
CREATE INDEX IF NOT EXISTS idx_ahash ON metadata(ahash);
CREATE INDEX IF NOT EXISTS idx_token_count ON metadata(token_count);
CREATE INDEX IF NOT EXISTS idx_bucket ON metadata(bucket);

CREATE TABLE IF NOT EXISTS schema_info (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


_VALID_COLUMNS = frozenset({
    "filename", "size_bytes", "mtime", "width", "height",
    "md5", "ahash", "tags", "token_count", "bucket", "cached_at",
})


class MetadataCache:
    """Thread-safe SQLite metadata cache for image datasets.

    Usage:
        cache = MetadataCache(dataset_dir / ".metadata.db")
        info = cache.get(image_path)
        if info is None or info["mtime"] != current_mtime:
            cache.put(image_path, {...})
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    @property
    def _conn(self) -> sqlite3.Connection:
        """Per-thread connection (SQLite is not thread-safe by default)."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path), timeout=10.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=-8192")  # 8 MB
        return self._local.conn

    def _init_db(self):
        """Create tables if they don't exist."""
        self._conn.executescript(_CREATE_SQL)
        # Check schema version
        row = self._conn.execute(
            "SELECT value FROM schema_info WHERE key='version'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO schema_info (key, value) VALUES ('version', ?)",
                (str(_SCHEMA_VERSION),),
            )
            self._conn.commit()

    def get(self, path: Path) -> Optional[dict]:
        """Get cached metadata for a file path. Returns None if not cached."""
        row = self._conn.execute(
            "SELECT * FROM metadata WHERE path = ?", (str(path),)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_if_fresh(self, path: Path) -> Optional[dict]:
        """Get cached metadata only if the file hasn't been modified since caching."""
        info = self.get(path)
        if info is None:
            return None
        try:
            st = os.stat(path)
            if abs(st.st_mtime - info["mtime"]) < 0.01 and st.st_size == info["size_bytes"]:
                return info
        except OSError:
            pass
        return None

    @staticmethod
    def _validate_columns(kwargs: dict) -> None:
        """Reject any column names not in the whitelist to prevent SQL injection."""
        invalid = set(kwargs.keys()) - _VALID_COLUMNS
        if invalid:
            raise ValueError(f"Invalid metadata column name(s): {invalid}")

    def put(self, path: Path, **kwargs):
        """Insert or update metadata for a file.

        Accepts keyword arguments matching column names:
            filename, size_bytes, mtime, width, height, md5, ahash,
            tags (comma-separated string), token_count, bucket, cached_at
        """
        import time
        kwargs.setdefault("filename", path.name)
        kwargs.setdefault("cached_at", time.time())
        self._validate_columns(kwargs)

        cols = ["path"] + list(kwargs.keys())
        placeholders = ", ".join(["?"] * len(cols))
        updates = ", ".join(f"{k}=excluded.{k}" for k in kwargs)

        self._conn.execute(
            f"INSERT INTO metadata ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(path) DO UPDATE SET {updates}",
            [str(path)] + list(kwargs.values()),
        )
        self._conn.commit()

    def put_batch(self, entries: list[tuple[Path, dict]]):
        """Bulk insert/update metadata entries for efficiency."""
        import time
        now = time.time()
        with self._conn:
            for path, kwargs in entries:
                kwargs.setdefault("filename", path.name)
                kwargs.setdefault("cached_at", now)
                self._validate_columns(kwargs)
                cols = ["path"] + list(kwargs.keys())
                placeholders = ", ".join(["?"] * len(cols))
                updates = ", ".join(f"{k}=excluded.{k}" for k in kwargs)
                self._conn.execute(
                    f"INSERT INTO metadata ({', '.join(cols)}) VALUES ({placeholders}) "
                    f"ON CONFLICT(path) DO UPDATE SET {updates}",
                    [str(path)] + list(kwargs.values()),
                )

    def query_by_tokens(self, min_tokens: int = 0, max_tokens: int = 999999) -> list[dict]:
        """Find images by token count range."""
        rows = self._conn.execute(
            "SELECT * FROM metadata WHERE token_count BETWEEN ? AND ? ORDER BY token_count DESC",
            (min_tokens, max_tokens),
        ).fetchall()
        return [dict(r) for r in rows]

    def query_duplicates(self, by: str = "md5") -> dict[str, list[str]]:
        """Find duplicate images grouped by hash.

        Args:
            by: Hash column to group by ('md5' or 'ahash')
        """
        if by not in ("md5", "ahash"):
            raise ValueError(f"Invalid hash column: {by}")
        rows = self._conn.execute(
            f"SELECT {by}, GROUP_CONCAT(path, '||') as paths, COUNT(*) as cnt "
            f"FROM metadata WHERE {by} IS NOT NULL "
            f"GROUP BY {by} HAVING cnt > 1"
        ).fetchall()
        result = {}
        for row in rows:
            result[row[0]] = row[1].split("||")
        return result

    def query_by_bucket(self, bucket: int) -> list[dict]:
        """Get all images assigned to a specific bucket."""
        rows = self._conn.execute(
            "SELECT * FROM metadata WHERE bucket = ?", (bucket,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Get aggregate statistics from the cache."""
        row = self._conn.execute("""
            SELECT
                COUNT(*) as total,
                AVG(token_count) as avg_tokens,
                MAX(token_count) as max_tokens,
                SUM(CASE WHEN token_count > 77 THEN 1 ELSE 0 END) as over_limit,
                COUNT(DISTINCT bucket) as num_buckets
            FROM metadata
        """).fetchone()
        return dict(row) if row else {}

    def has_entry(self, path: Path) -> bool:
        """Check if a path exists in the cache."""
        row = self._conn.execute(
            "SELECT 1 FROM metadata WHERE path = ? LIMIT 1", (str(path),)
        ).fetchone()
        return row is not None

    def remove(self, path: Path):
        """Remove a single entry from the cache."""
        self._conn.execute("DELETE FROM metadata WHERE path = ?", (str(path),))
        self._conn.commit()

    def clear(self):
        """Clear all cached metadata."""
        self._conn.execute("DELETE FROM metadata")
        self._conn.commit()

    def count(self) -> int:
        """Return the number of cached entries."""
        row = self._conn.execute("SELECT COUNT(*) FROM metadata").fetchone()
        return row[0] if row else 0

    def close(self):
        """Close the database connection for the current thread."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
