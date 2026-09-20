"""Regenerate with torch 2.14.0 and transformers 4.57.6 (CPU, seed 42)."""
import json
from pathlib import Path
import torch
from transformers import DebertaV2Config, DebertaV2Model

torch.set_num_threads(1)
root = Path(__file__).resolve().parent
variants = {
    "absolute": {},
    "shared": dict(relative_attention=True, pos_att_type=["c2p", "p2c"], share_att_key=True,
                   position_buckets=8, norm_rel_ebd="layer_norm", position_biased_input=False),
    "separate_conv": dict(relative_attention=True, pos_att_type=["c2p", "p2c"],
                          position_buckets=8, conv_kernel_size=3, conv_groups=3,
                          embedding_size=8, type_vocab_size=2),
    "content_to_position": dict(relative_attention=True, pos_att_type=["c2p"], max_relative_positions=6),
    "position_to_content": dict(relative_attention=True, pos_att_type=["p2c"]),
}
for name, options in variants.items():
    torch.manual_seed(42)
    config = DebertaV2Config(vocab_size=32, hidden_size=12, num_hidden_layers=2,
                            num_attention_heads=3, intermediate_size=17,
                            max_position_embeddings=32, **options)
    model = DebertaV2Model(config).eval()
    inputs = {"input_ids": torch.tensor([[2,3,4,5,6,7,8,9,10,11,12,13], [9,8,7,6,5,4,3,2,0,0,0,0]]),
              "attention_mask": torch.tensor([[1]*12, [1]*8+[0]*4])}
    if config.type_vocab_size:
        inputs["token_type_ids"] = torch.tensor([[0]*6+[1]*6, [1]*8+[0]*4])
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True, output_attentions=True)
        unmasked = model(input_ids=inputs["input_ids"])
    directory = root / name
    model.save_pretrained(directory)
    data = {"inputs": {k:v.tolist() for k,v in inputs.items()},
            "hidden_state": output.last_hidden_state.tolist(),
            "hidden_states": [v.tolist() for v in output.hidden_states],
            "attentions": [v.tolist() for v in output.attentions],
            "unmasked": unmasked.last_hidden_state.tolist()}
    (directory / "expected.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")

# Small Unigram tokenizer exercises the DeBERTa special-token defaults.
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
raw = Tokenizer(models.Unigram([(s, -1.0) for s in ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "▁hello", "▁world"]], unk_id=3))
raw.pre_tokenizer = pre_tokenizers.Metaspace()
raw.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]",1),("[SEP]",2)])
tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]", mask_token="[MASK]")
directory = root / "tokenizer"
tokenizer.save_pretrained(directory)
(directory / "config.json").write_text(json.dumps({"model_type":"deberta-v2"}) + "\n")
(directory / "expected.json").write_text(json.dumps(tokenizer("hello world").data) + "\n")
