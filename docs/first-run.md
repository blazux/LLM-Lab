# Your first graphical experiment

Install and launch using [the installation guide](installation.md), then open the
local address printed by `bash gui/start.sh`. Keep the terminal open. Python
3.11–3.13 is supported; select the CUDA environment to use your NVIDIA GPU.

1. Create a **Pretrain** project. Open **Architecture templates**, choose
   **Llama 3 / Nano**, and build its editable graph. This creates random weights,
   not a downloaded language model. Keep the tokenizer setting or select your
   own tokenizer; the workshop reads its actual vocabulary before training.
2. Follow the arrows from Token IDs at the bottom to LM Head at the top. Open a
   repeated block to find RMSNorm, Self-Attention, residual additions and FFN.
   Try changing the FFN size: the parameter count must change. Undo restores it.
   Moving a brick changes the drawing; reconnecting it changes the computation.
3. Run the workshop's forward/backward probe. A green result checks that the
   graph computes and differentiates; it does not certify language quality.
4. In **Dataset**, use the beginner TinyStories guide. It shows the actual source
   and held-out document ranges. Choose the short test settings first. Read the
   dataset card and license before a larger run. Downloads require connectivity.
5. In **Training**, inspect the optimizer-update and approximate token budget.
   Start with the suggested small batch. Accumulation lets several microbatches
   contribute to one update. Leave optional compilation and experimental losses
   disabled for the first run.
6. In **Run**, start training. Inspect training objective, language cross-entropy
   and validation perplexity. A noisy curve or short plateau is not proof that a
   model has stopped learning. Validation and generated samples matter together.
7. Stop the run and wait for **stopped**. Reload the page: its history remains.
   Resume its saved checkpoint into a new run folder. Set additional optimizer
   steps if its original budget is already complete.
8. In **Model Tools**, choose that recorded run and its checkpoint. Try
   **Playground**, then **Export locally**. Inspect the output model card,
   tokenizer and executable model files. For sharing, use **Export & Push** with
   your Hub repository. Private is the default; the backend can use `hf auth login`
   or a write token entered for that request.

A few updates demonstrate the workflow, not useful language. Learning a language
model needs a coherent corpus, many tokens and validation. Keep early experiments
small, preserve a baseline, and change one idea at a time. See the
[dataset guide](datasets.md), [recorded run guide](minimal-pretraining.md#recorded-runs-in-the-interface)
and [capability matrix](capabilities.md) for the supported scope.
