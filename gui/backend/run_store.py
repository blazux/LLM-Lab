"""Durable local run registry and replayable events. One API process owns training."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
LEGACY_METRICS = 'objective_exp_v1'
LANGUAGE_METRICS = 'language_cross_entropy_v2'


def data_directory():
    return Path(os.environ.get('LLMLAB_DATA_DIR', '/app/data' if Path('/app').exists() else ROOT / 'data')).resolve()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


class RunStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'runs.sqlite3'
        self.lock = threading.RLock()
        with self.connection() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, project_id TEXT, output_dir TEXT NOT NULL,
                    created_at TEXT NOT NULL, config TEXT NOT NULL, state TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    data TEXT NOT NULL, FOREIGN KEY(run_id) REFERENCES runs(id)
                );
                CREATE INDEX IF NOT EXISTS events_by_run ON events(run_id, id);
            ''')

    @contextmanager
    def connection(self):
        with self.lock:
            db = sqlite3.connect(self.path, timeout=30)
            db.row_factory = sqlite3.Row
            try:
                with db:
                    yield db
            finally:
                db.close()

    def create(self, state, project_id=None):
        run_id = uuid.uuid4().hex
        state = {**state, 'run_id': run_id}
        config = {key: state.pop(key, None) for key in ('model_config', 'training_config')}
        output = str(Path(state.get('output_dir') or self.directory / 'runs' / run_id).resolve())
        Path(output).mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute('INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)', (
                run_id, project_id, output, datetime.now(timezone.utc).isoformat(),
                json.dumps(clean(config)), json.dumps(clean(state))))
        return self.get(run_id)

    def get(self, run_id):
        with self.connection() as db:
            row = db.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return {**json.loads(row['state']), **json.loads(row['config']), 'run_id': row['id'],
                'project_id': row['project_id'], 'output_dir': row['output_dir'], 'created_at': row['created_at']}

    def list(self, project_id=None):
        with self.connection() as db:
            rows = db.execute('SELECT id FROM runs ORDER BY created_at DESC, rowid DESC').fetchall()
        result = [self.get(row['id']) for row in rows]
        return [row for row in result if project_id is None or row['project_id'] == project_id]

    def update(self, run_id, changes):
        changes = clean(changes)
        with self.connection() as db:
            row = db.execute('SELECT state, config FROM runs WHERE id=?', (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            state, config = json.loads(row['state']), json.loads(row['config'])
            if 'project_id' in changes:
                db.execute('UPDATE runs SET project_id=? WHERE id=?', (changes.pop('project_id'), run_id))
            for key, value in changes.items():
                (config if key in config else state)[key] = value
            db.execute('UPDATE runs SET state=?, config=? WHERE id=?',
                       (json.dumps(state), json.dumps(config), run_id))

    def append(self, run_id, event):
        event = clean({'timestamp': time.time(), **event, 'run_id': run_id})
        with self.connection() as db:
            cursor = db.execute('INSERT INTO events(run_id,data) VALUES (?,?)', (run_id, json.dumps(event)))
            return {**event, 'event_id': cursor.lastrowid}

    def events(self, run_id, after=0, limit=1000):
        self.get(run_id)
        with self.connection() as db:
            rows = db.execute('SELECT id, data FROM events WHERE run_id=? AND id>? ORDER BY id LIMIT ?',
                              (run_id, after, limit)).fetchall()
        return [{**json.loads(row['data']), 'event_id': row['id']} for row in rows]

    def recover(self):
        """An API restart never claims that a dead daemon thread is still training."""
        for run in self.list():
            if run.get('is_training'):
                self.update(run['run_id'], {'is_training': False, 'status': 'interrupted',
                    'error': 'The API stopped before this run finished. Resume from a saved checkpoint.'})

    def checkpoint_files(self, run_id=None):
        records = [self.get(run_id)] if run_id else self.list()
        if run_id:
            roots = {Path(records[0]['output_dir'])}
        else:
            roots = {self.directory, Path('/app/data')}
            roots.update(Path(run['output_dir']) for run in records)
        roots.update(Path(run['output_dir']) / 'experiment' for run in records
                     if run.get('training_type') == 'learning')
        files = set()
        for root in roots:
            for folder in (root, root / 'checkpoints'):
                if folder.is_dir():
                    files.update(path.resolve() for path in folder.glob('*.pt') if path.is_file())
        return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


_default = None
_default_lock = threading.Lock()


def get_run_store():
    global _default
    with _default_lock:
        if _default is None:
            _default = RunStore(data_directory())
            _default.recover()
        return _default
