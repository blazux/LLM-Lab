# Small-model learning corpus

These 384 short, synthetic English documents are provided under the repository’s
MIT license. They combine twelve names, eight objects and four actions. The
training split has 320 documents; validation has 64 different documents. Entire
sentences do not occur in both splits. The same vocabulary and sentence patterns
are shared deliberately: validation measures prediction on held-out combinations,
not general knowledge or language competence.

The tokenizer is fitted on training documents only. No remote dataset or model
is required. The experiment records both file hashes and prepared document counts.
