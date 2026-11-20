"""Source releases include the learning assets and exclude private working records."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('source_release_builder', ROOT/'scripts/build_release.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class SourceDistributionTests(unittest.TestCase):
    def test_private_records_and_generated_data_never_enter_source_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = ['docs/learning-path.md', 'docs/residual-connections.md',
                      'recipes/learning-data/train.txt', 'recipes/learning-data/validation.txt',
                      'gui/frontend/src/content/modern-bricks.json']
            private = ['docs/roadmap-1.0.md', 'docs/release-validation.md',
                       'docs/assets/reference-learning.png', 'docs/release-evidence/local.json',
                       'docs/release-evidence/nested/profile.json',
                       'gui/backend/data/runs/private.json', 'gui/frontend/node_modules/pkg/index.js',
                       'gui/frontend/dist/index.html', 'src/cache/local.json', '.env']
            for name in public + private:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            with patch.object(builder, 'ROOT', root):
                included = {path.relative_to(root).as_posix() for path in builder.sources()}
            self.assertEqual(included, set(public))

    def test_learning_experience_sources_are_distributed(self):
        included = {path.relative_to(ROOT).as_posix() for path in builder.sources()}
        for name in ['gui/backend/api/learning.py', 'recipes/small-model-learning.toml',
                     'recipes/residual-connections.toml', 'recipes/learning-data/train.txt',
                     'recipes/learning-data/validation.txt', 'docs/learning-path.md',
                     'docs/residual-connections.md',
                     'gui/frontend/src/components/LearningExperiment.tsx',
                     'gui/frontend/src/components/LearningRunComparison.tsx',
                     'gui/frontend/src/components/ResidualDiagram.tsx']:
            with self.subTest(source=name):
                self.assertIn(name, included)

    def test_symlinks_do_not_add_external_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'source'
            docs = root/'docs'
            docs.mkdir(parents=True)
            outside = Path(tmp)/'external'
            outside.mkdir()
            (outside/'record.md').write_text('fixture')
            (docs/'linked.md').symlink_to(outside/'record.md')
            (docs/'linked-directory').symlink_to(outside, target_is_directory=True)
            with patch.object(builder, 'ROOT', root):
                self.assertEqual(builder.sources(), [])


if __name__ == '__main__':
    unittest.main()
