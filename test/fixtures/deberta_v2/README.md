# DeBERTa reference fixtures

`generate.py` creates small checkpoints and reference outputs using the
Transformers `DebertaV2Model`, plus a small Unigram tokenizer. The generated
files are committed, so running the Elixir tests requires neither Python nor
a checkpoint download. Python is only needed to regenerate the fixtures.

## Regeneration

The fixtures were generated with Python **3.12.13**, torch **2.14.0**,
transformers **4.57.6**, safetensors **0.8.0**, and tokenizers **0.22.2**.
From the Bumblebee repository root:

```sh
python3.12 -m venv tmp/deberta-fixtures-venv
tmp/deberta-fixtures-venv/bin/python -m pip install \
  'torch==2.14.0' 'transformers==4.57.6' \
  'safetensors==0.8.0' 'tokenizers==0.22.2'
tmp/deberta-fixtures-venv/bin/python test/fixtures/deberta_v2/generate.py
mix test test/bumblebee/text/deberta_v2_test.exs
git diff --stat -- test/fixtures/deberta_v2
```

The virtual environment is under the repository's ignored `tmp/` directory.
Installing the packages requires network access; the generator itself does not
download pretrained models. It runs on CPU with one PyTorch thread, resets
`torch.manual_seed(42)` before each model, and uses evaluation and inference
modes. Numerical outputs may differ slightly across platforms even with the
same seed and package versions; review regenerated values rather than assuming
byte-identical results.

## Generated files

Each model directory contains `config.json`, `model.safetensors`, and
`expected.json`:

| Directory | Coverage |
| --- | --- |
| `absolute/` | Absolute positions without relative attention |
| `shared/` | Shared relative projections, both relative-attention directions, buckets, and normalized relative embeddings |
| `separate_conv/` | Separate relative projections, grouped convolution, embedding projection, and token types |
| `content_to_position/` | Content-to-position attention with an explicit relative-position limit |
| `position_to_content/` | Position-to-content attention |

Each `expected.json` records the inputs, final hidden state, intermediate hidden
states, attention outputs, and the output when the attention mask is omitted.
The inputs include a padded sequence to exercise masking.

The `tokenizer/` directory contains `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `config.json`, and `expected.json`. These describe a
small Unigram vocabulary and the expected token IDs and mask for `hello world`,
including the DeBERTa special tokens.

Regeneration overwrites these files. No GLiNER fixtures or packages are needed
to generate or test the DeBERTa fixtures.
