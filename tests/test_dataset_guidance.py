"""Offline checks for the dataset controls used by guided pretraining presets."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data.data import create_token_stream, validate_dataset_separation


class Tokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return [int(text)]


class DatasetGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'corpus.jsonl'
        self.path.write_text(''.join(json.dumps({'content': str(i), 'metadata': i}) + '\n' for i in range(10)))
        self.source = dict(name='json', data_files=str(self.path), split='train', text_field='content')

    def test_document_holdout_is_disjoint_and_reiterable(self):
        train = [{**self.source, 'skip': 3, 'take': 4}]
        val = [{**self.source, 'take': 3}]
        validate_dataset_separation(train, val)
        stream = create_token_stream(train, Tokenizer())
        self.assertEqual(list(stream()), [[i, 99] for i in range(3, 7)])
        self.assertEqual(list(stream()), list(stream()))
        self.assertEqual(list(create_token_stream(val, Tokenizer())()), [[i, 99] for i in range(3)])

    def test_revision_is_forwarded_to_hub_loader(self):
        from datasets import IterableDataset
        rows = IterableDataset.from_generator(lambda: iter([{'text': '7'}]))
        with patch('data.data.load_dataset', return_value=rows) as loader:
            stream = create_token_stream([dict(name='org/corpus', subset='english', split='validation', revision='pinned')], Tokenizer())
            self.assertEqual(list(stream()), [[7, 99]])
        loader.assert_called_once_with('org/corpus', name='english', split='validation', streaming=True, revision='pinned')

    def test_mixed_text_columns_are_normalized_before_interleaving(self):
        second = Path(self.tmp.name) / 'other.jsonl'
        second.write_text(json.dumps({'text': '42'}) + '\n')
        stream = create_token_stream([self.source, dict(name='json', data_files=str(second), weight=1)], Tokenizer())
        values = list(stream())
        self.assertTrue(values)
        self.assertTrue(all(len(doc) == 2 and doc[-1] == 99 for doc in values))

    def test_missing_source_is_not_silently_omitted_from_mixture(self):
        with self.assertRaises(Exception):
            create_token_stream([self.source, dict(name='json', data_files=str(self.path.parent / 'missing.jsonl'))], Tokenizer())

    def test_invalid_ranges_and_weights_fail(self):
        for bad in [dict(skip=-1), dict(skip=0.5), dict(take=0), dict(take=True), dict(weight=0), dict(weight=float('nan'))]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                create_token_stream([{**self.source, **bad}], Tokenizer())

    def test_overlap_is_rejected_even_across_revisions(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_dataset_separation([{**self.source, 'skip': 2, 'take': 4, 'revision': 'a'}], [{**self.source, 'skip': 5, 'revision': 'b'}])
        validate_dataset_separation([{**self.source, 'skip': 3}], [{**self.source, 'take': 3}])
        validate_dataset_separation([self.source], [{**self.source, 'split': 'validation'}])

    def test_wrong_text_column_fails_instead_of_empty_training(self):
        with self.assertRaises((ValueError, KeyError)):
            list(create_token_stream([{**self.source, 'text_field': 'absent'}], Tokenizer())())


if __name__ == '__main__':
    unittest.main()
