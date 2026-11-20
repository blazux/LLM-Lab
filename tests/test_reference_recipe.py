"""Regression checks for document separation in the reference experiment."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from reference_pretraining import prepare_corpus, TRAIN_FILE, VALID_FILE


class ReferenceCorpusTests(unittest.TestCase):
    def test_normalized_duplicates_cannot_cross_validation_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            (source / 'data').mkdir(parents=True)
            output = root / 'prepared'
            output.mkdir()
            pq.write_table(pa.table({'text': ['A small story.', 'A  small story.', 'Another story.']}),
                           source / 'data' / TRAIN_FILE)
            pq.write_table(pa.table({'text': [' A small story. ', 'Validation story.', 'Validation  story.', 'Other validation.']}),
                           source / 'data' / VALID_FILE)
            result = prepare_corpus(source, output, 2, 2)
            train = [json.loads(line)['text'] for line in (output / 'train.jsonl').read_text().splitlines()]
            val = [json.loads(line)['text'] for line in (output / 'validation.jsonl').read_text().splitlines()]
            self.assertEqual(train, ['A small story.', 'Another story.'])
            self.assertEqual(val, ['Validation story.', 'Other validation.'])
            self.assertEqual(result['train']['duplicates_skipped'], 1)
            self.assertEqual(result['validation']['duplicates_skipped'], 2)
            self.assertEqual(len(result['validation']['sha256']), 64)

    def test_insufficient_unique_data_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data').mkdir()
            pq.write_table(pa.table({'text': ['', 'only story', 'only story']}), root / 'data' / TRAIN_FILE)
            output = root / 'prepared'
            output.mkdir()
            with self.assertRaisesRegex(ValueError, 'unique train documents'):
                prepare_corpus(root, output, 2, 1)


if __name__ == '__main__':
    unittest.main()
