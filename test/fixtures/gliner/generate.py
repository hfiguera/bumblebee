"""Generate offline head references with gliner2 2.0.0 / torch 2.14.0."""
import dataclasses
import json
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
from gliner2.configuration import BoundaryHeadSettings
from gliner2.models.boundary.model import BoundaryHead
from gliner2.models.boundary.pool import PooledCandidates

root = Path(__file__).resolve().parent
torch.set_num_threads(1)
for name, layers, window in [("local", 1, 2), ("global", 1, 0), ("no_refinement", 0, 0)]:
    torch.manual_seed(42)
    settings = BoundaryHeadSettings(boundary_dim=8, pair_dim=8, content_dim=4,
        boundary_attention_heads=2, boundary_attention_layers=layers,
        boundary_refinement_layers=layers, boundary_attention_window=window,
        candidate_pool="shared", candidate_attention_layers=0, query_attention_layers=0,
        content_soft_max_pool=False, enable_span_content=True, enable_abstention=True,
        use_inside_evidence=True, adaptive_threshold=False, overlap_policy="flat",
        pool_size=8, pool_boundary_top_k=4, min_pool_per_query=2)
    head = BoundaryHead(12, settings).eval()
    text = torch.randn(1, 7, 12)
    query = torch.randn(1, 3, 12)
    text_mask = torch.ones(1,7,dtype=torch.bool)
    query_mask = torch.ones(1,3,dtype=torch.bool)
    with torch.inference_mode():
        boundary = head.boundary_encoder(text, text_mask)
        margins = head.boundary_query_head(boundary.states, boundary.mask, text, text_mask, query, query_mask)
        pooled = head.shared_pool_builder(boundary.states, boundary.mask, query_mask, margins.start_logits, margins.end_logits)
        # Retain a padded candidate to check score masking.
        pooled.mask[:, -1] = False
        scores, _ = head.shared_pool_scorer(boundary.states, query, query_mask, pooled,
            margins.start_logits, margins.end_logits, margins.inside_prefix,
            torch.tensor([7]), text, text_mask, margins.inside_prefix_mean)
        output = {"boundary_states": boundary.states, "start_logits": margins.start_logits,
                  "end_logits": margins.end_logits, "inside_logits": margins.inside_logits,
                  "null_logits": head.null_projection(query),
                  "pool_start": head.shared_pool_builder.start_projection(boundary.states),
                  "pool_end": head.shared_pool_builder.end_projection(boundary.states)}
    inputs = {"text_states": text, "query_states": query}
    scorer_inputs = {**inputs, **{k:output[k] for k in ["boundary_states","start_logits","end_logits","inside_logits"]},
                     "starts": pooled.indices[...,0], "ends": pooled.indices[...,1],
                     "valid": pooled.mask.to(torch.uint8), "compat": pooled.compat_logits}
    directory = root / name
    directory.mkdir(exist_ok=True)
    state = {"boundary_head."+key: value for key,value in head.state_dict().items()}
    if name == "local":
        state.update({"encoder."+key:value for key,value in load_file(root.parent / "deberta_v2/shared/model.safetensors").items()})
    save_file(state, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps({"architectures":["BoundaryExtractor"], "architecture":"boundary", "token_pooling":"first", "boundary_head":dataclasses.asdict(settings)}, indent=2)+"\n")
    (directory / "expected.json").write_text(json.dumps({"inputs":{k:v.tolist() for k,v in inputs.items()},"outputs":{k:v.tolist() for k,v in output.items()},"scorer_inputs":{k:v.tolist() for k,v in scorer_inputs.items()},"scores":scores.transpose(1,2).tolist()},separators=(",",":"))+"\n")
