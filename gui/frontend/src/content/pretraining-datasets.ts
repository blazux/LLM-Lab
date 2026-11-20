import type { DatasetEntry } from '../types'

export interface CorpusChoice {
  id: string
  title: string
  language: string
  purpose: string
  size: string
  quality: string
  limitation: string
  license: string
  source: DatasetEntry
  validationSplit?: string
}

// Dataset cards and Hub configurations checked on 2026-10-16.
// Published corpus sizes are context, never the amount a run must consume.
export const PRETRAINING_CORPORA: CorpusChoice[] = [
  {
    id: 'stories', title: 'TinyStories', language: 'English', purpose: 'My first experiment',
    size: 'About 2.1M training stories', license: 'CDLA-Sharing 1.0',
    quality: 'Short synthetic stories with a simple vocabulary. A small model can learn recognizable story completion.',
    limitation: 'Narrow vocabulary and subject matter. Success here does not imply general knowledge or assistant skills.',
    source: { name: 'roneneldan/TinyStories', split: 'train', revision: 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64' },
    validationSplit: 'validation',
  },
  {
    id: 'web-edu', title: 'FineWeb-Edu', language: 'English', purpose: 'General language and knowledge',
    size: '10B-token sample selected; full release about 1.3T tokens', license: 'ODC-By 1.0',
    quality: 'Web pages selected by an educational-quality classifier. A useful starting point for a broader English model.',
    limitation: 'Educational scores are a proxy for quality. Web noise remains; meaningful general capabilities require much more compute than a short test.',
    source: { name: 'HuggingFaceFW/fineweb-edu', subset: 'sample-10BT', split: 'train', revision: '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9' },
  },
  {
    id: 'french', title: 'FineWeb2 · French', language: 'French', purpose: 'Learn from French text',
    size: 'Large French web corpus; streamed on demand', license: 'ODC-By 1.0',
    quality: 'Language-specific filtering and deduplication of web text. The fra_Latn configuration selects French.',
    limitation: 'Broad web text, not an educational-only selection. Use a tokenizer that represents French efficiently.',
    source: { name: 'HuggingFaceFW/fineweb-2', subset: 'fra_Latn', split: 'train', revision: 'af9c13333eb981300149d5ca60a8e9d659b276b9' },
  },
  {
    id: 'textbooks', title: 'Cosmopedia v2', language: 'English', purpose: 'Educational synthetic text',
    size: 'About 39M documents', license: 'ODC-By 1.0 (repository metadata)',
    quality: 'Synthetic textbooks, stories and articles used in the SmolLM corpus. Useful for studying an educational-data mixture.',
    limitation: 'Generated text can repeat patterns and factual errors. It is not a verified reference library.',
    source: { name: 'HuggingFaceTB/smollm-corpus', subset: 'cosmopedia-v2', split: 'train', revision: '3ba9d605774198c5868892d7a8deda78031a781f' },
  },
  {
    id: 'math', title: 'OpenWebMath', language: 'English', purpose: 'Explore mathematical text',
    size: 'About 14.7B tokens', license: 'ODC-By 1.0; underlying content licenses still apply',
    quality: 'Filtered mathematical web text with LaTeX preserved. Useful as a specialist corpus or mixture component.',
    limitation: 'Not a set of verified solutions. Start with a general corpus before experimenting with sampling weights.',
    source: { name: 'open-web-math/open-web-math', split: 'train', revision: 'fde8ef8de2300f5e778f56261843dab89f230815' },
  },
]

export function corpusSources(choice: CorpusChoice, trainDocuments?: number) {
  const base = { ...choice.source, text_field: 'text' }
  return {
    datasets: [{ ...base, skip: choice.validationSplit ? 0 : 512, ...(trainDocuments ? { take: trainDocuments } : {}) }],
    validation_datasets: [{ ...base, split: choice.validationSplit ?? base.split, take: 512 }],
  }
}
