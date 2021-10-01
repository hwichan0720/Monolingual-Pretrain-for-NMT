import math
import copy
import torch
from torch import Tensor
from fairseq.models.bart import BARTModel
from fairseq.data import Dictionary
from typing import List, Dict, Optional


def load_bart(path, model_name):
    bart = BARTModel.from_pretrained(path, checkpoint_file=model_name)
    bart.eval()
    return bart

def load_sents(file_path):
    with open(file_path, 'r') as f:
        sents = f.readlines()
    return [sent.strip() for sent in sents]


def load_dict(path: str) -> Dictionary:
    d = Dictionary.load(path)
    # for l in langs:
    d.add_symbol("<mask>")
    return d

def sent_to_ids(d, sent, max_length=128):
    tokens = sent.split()
    return tokens_to_ids(d, tokens, max_length)

def tokens_to_ids(d, tokens, max_length=128):
    ids = [d.index('<pad>') for _ in range(max_length)]
    ids = torch.tensor(ids)
    for n, token in enumerate(tokens):
        id_ = d.index(token)
        ids[n] = id_
    return ids

def ids_to_tokens(d, idxs):
    tokens = []
    for idx in idxs:
        token = d[idx]
        tokens.append(token)
    return tokens

def _forward_layer(
    layer,
    x,
    encoder_out: Optional[torch.Tensor] = None,
    encoder_padding_mask: Optional[torch.Tensor] = None,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    prev_self_attn_state: Optional[List[torch.Tensor]] = None,
    prev_attn_state: Optional[List[torch.Tensor]] = None,
    self_attn_mask: Optional[torch.Tensor] = None,
    self_attn_padding_mask: Optional[torch.Tensor] = None,
    need_attn: bool = False,
    need_head_weights: bool = False,
):
    if need_head_weights:
            need_attn = True

    residual = x
    if layer.normalize_before:
        x = layer.self_attn_layer_norm(x)
    if prev_self_attn_state is not None:
        prev_key, prev_value = prev_self_attn_state[:2]
        saved_state: Dict[str, Optional[Tensor]] = {
            "prev_key": prev_key,
            "prev_value": prev_value,
        }
        if len(prev_self_attn_state) >= 3:
            saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
        assert incremental_state is not None
        layer.self_attn._set_input_buffer(incremental_state, saved_state)
    _self_attn_input_buffer = layer.self_attn._get_input_buffer(incremental_state)
    if layer.cross_self_attention and not (
        incremental_state is not None
        and _self_attn_input_buffer is not None
        and "prev_key" in _self_attn_input_buffer
    ):
        if self_attn_mask is not None:
            assert encoder_out is not None
            self_attn_mask = torch.cat(
                (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
            )
        if self_attn_padding_mask is not None:
            if encoder_padding_mask is None:
                assert encoder_out is not None
                encoder_padding_mask = self_attn_padding_mask.new_zeros(
                    encoder_out.size(1), encoder_out.size(0)
                )
            self_attn_padding_mask = torch.cat(
                (encoder_padding_mask, self_attn_padding_mask), dim=1
            )
        assert encoder_out is not None
        y = torch.cat((encoder_out, x), dim=0)
    else:
        y = x

    x, attn = layer.self_attn(
        query=x,
        key=y,
        value=y,
        key_padding_mask=self_attn_padding_mask,
        incremental_state=incremental_state,
        need_weights=False,
        attn_mask=self_attn_mask,
    )
    x = layer.dropout_module(x)
    x = residual + x
    if not layer.normalize_before:
        x = layer.self_attn_layer_norm(x)
        
    self_dec_attn = copy.copy(x)

    if layer.encoder_attn is not None and encoder_out is not None:
        residual = x
        if layer.normalize_before:
            x = layer.encoder_attn_layer_norm(x)
        if prev_attn_state is not None:
            prev_key, prev_value = prev_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_attn_state[2]
            assert incremental_state is not None
            layer.encoder_attn._set_input_buffer(incremental_state, saved_state)

        x, attn = layer.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=need_attn or (not layer.training and layer.need_attn),
            need_head_weights=need_head_weights,
        )
        x = layer.dropout_module(x)
        x = residual + x
        if not layer.normalize_before:
            x = layer.encoder_attn_layer_norm(x)

    residual = x
    if layer.normalize_before:
        x = layer.final_layer_norm(x)

    x = layer.activation_fn(layer.fc1(x))
    x = layer.activation_dropout_module(x)
    x = layer.fc2(x)
    x = layer.dropout_module(x)
    x = residual + x
    if not layer.normalize_before:
        x = layer.final_layer_norm(x)
    if layer.onnx_trace and incremental_state is not None:
        saved_state = layer.self_attn._get_input_buffer(incremental_state)
        assert saved_state is not None
        if self_attn_padding_mask is not None:
            self_attn_state = [
                saved_state["prev_key"],
                saved_state["prev_value"],
                saved_state["prev_key_padding_mask"],
            ]
        else:
            self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
        return x, attn, self_attn_state, None
    return x, attn, self_dec_attn, None

def _forward_decoder(
    decoder,
    prev_output_tokens,
    encoder_out,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    full_context_alignment: bool = False,
    alignment_layer: Optional[int] = None,
    alignment_heads: Optional[int] = None,
):
    if alignment_layer is None:
        alignment_layer = decoder.num_layers - 1

    # embed positions
    positions = (
        decoder.embed_positions(
            prev_output_tokens, incremental_state=incremental_state
        )
        if decoder.embed_positions is not None
        else None
    )

    if incremental_state is not None:
        prev_output_tokens = prev_output_tokens[:, -1:]
        if positions is not None:
            positions = positions[:, -1:]

    # embed tokens and positions
    x = decoder.embed_scale * decoder.embed_tokens(prev_output_tokens)

    if decoder.quant_noise is not None:
        x = decoder.quant_noise(x)

    if decoder.project_in_dim is not None:
        x = decoder.project_in_dim(x)

    if positions is not None:
        x += positions

    if decoder.layernorm_embedding is not None:
        x = decoder.layernorm_embedding(x)

    x = decoder.dropout_module(x)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    self_attn_padding_mask: Optional[Tensor] = None
    if decoder.cross_self_attention or prev_output_tokens.eq(decoder.padding_idx).any():
        self_attn_padding_mask = prev_output_tokens.eq(decoder.padding_idx)

    # decoder layers
    attn: Optional[Tensor] = None
    inner_states: List[Optional[Tensor]] = [x]
    self_dec_attns = []
    for idx, layer in enumerate(decoder.layers):
        if incremental_state is None and not full_context_alignment:
            self_attn_mask = decoder.buffered_future_mask(x)
        else:
            self_attn_mask = None

        x, layer_attn, self_dec_attn, _ = _forward_layer(
            layer,
            x,
            encoder_out.encoder_out if encoder_out is not None else None,
            encoder_out.encoder_padding_mask if encoder_out is not None else None,
            incremental_state,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_attn=bool((idx == alignment_layer)),
            need_head_weights=bool((idx == alignment_layer)),
        )
        inner_states.append(x)
        self_dec_attns.append(self_dec_attn)
        if layer_attn is not None and idx == alignment_layer:
            attn = layer_attn.float().to(x)

    if attn is not None:
        if alignment_heads is not None:
            attn = attn[:alignment_heads]

        # average probabilities over heads
        attn = attn.mean(dim=0)

    if decoder.layer_norm is not None:
        x = decoder.layer_norm(x)

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    if decoder.project_out_dim is not None:
        x = decoder.project_out_dim(x)
    
    x = decoder.output_layer(x)

    return x, {"attn": [attn], "inner_states": inner_states, "self_dec_attns": self_dec_attns}

def forward_decoder(
    ensemble_model,
    tokens,
    encoder_outs,
    incremental_states,
    temperature: float = 1.0,
):
    log_probs = []
    probs = []
    avg_attn: Optional[Tensor] = None
    encoder_out: Optional[EncoderOut] = None
    for i, model in enumerate(ensemble_model.models):
        if ensemble_model.has_encoder():
            encoder_out = encoder_outs[i]
        # decode each model
        if ensemble_model.has_incremental_states():
            decoder_out = _forward_decoder(
                model.decoder,
                tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_states[i],
            )
            # decoder_out = model.decoder.forward(
            #     tokens,
            #     encoder_out=encoder_out,
            #     incremental_state=incremental_states[i],
            #     return_all_hiddens=True
            # )

        else:
            decoder_out = _forward_decoder(model.decoder, tokens, encoder_out=encoder_out)
            # decoder_out = model.decoder.forward(
            #     tokens, 
            #     encoder_out=encoder_out,
            #     return_all_hiddens=True
            # )

        attn: Optional[Tensor] = None
        decoder_len = len(decoder_out)
        if decoder_len > 1 and decoder_out[1] is not None:
            if isinstance(decoder_out[1], Tensor):
                attn = decoder_out[1]
            else:
                attn_holder = decoder_out[1]["attn"]
                if isinstance(attn_holder, Tensor):
                    attn = attn_holder
                elif attn_holder is not None:
                    attn = attn_holder[0]
            if attn is not None:
                attn = attn[:, -1, :]

        decoder_out_tuple = (
            decoder_out[0][:, -1:, :].div_(temperature),
            None if decoder_len <= 1 else decoder_out[1],
        )

        log_prob = model.get_normalized_probs(
            decoder_out_tuple, log_probs=True, sample=None
        )        
        log_prob = log_prob[:, -1, :]
        log_probs.append(log_prob)
        
        prob = model.get_normalized_probs(
            decoder_out_tuple, log_probs=False, sample=None
        )        
        prob = prob[:, -1, :]
        probs.append(prob)
        
        if ensemble_model.models_size == 1:
            return decoder_out, log_prob, prob, attn
        if attn is not None:
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
    avg_log_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
        ensemble_model.models_size
    )
    avg_probs = torch.logsumexp(torch.stack(probs, dim=0), dim=0) - math.log(
        ensemble_model.models_size
    )
    
    if avg_attn is not None:
        avg_attn.div_(ensemble_model.models_size)
    return decoder_out, avg_log_probs, avg_probs, avg_attn


def _generate(
    generator,
    sample,
    prefix_tokens=None,
    constraints=None,
    bos_token=None,
):
    incremental_states = torch.jit.annotate(
        List[Dict[str, Dict[str, Optional[Tensor]]]],
        [
            torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
            for i in range(generator.model.models_size)
        ],
    )
    net_input = sample["net_input"]

    if 'src_tokens' in net_input:
        src_tokens = net_input['src_tokens']
        # length of the source text being the character length except EndOfSentence and pad
        src_lengths = (src_tokens.ne(generator.eos) & src_tokens.ne(generator.pad)).long().sum(dim=1)
    elif 'source' in net_input:
        src_tokens = net_input['source']
        src_lengths = (
            net_input['padding_mask'].size(-1) - net_input['padding_mask'].sum(-1)
            if net_input['padding_mask'] is not None
            else torch.tensor(src_tokens.size(-1)).to(src_tokens)
        )
    else:
        raise Exception('expected src_tokens or source in net input')

    # bsz: total number of sentences in beam
    # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
    bsz, src_len = src_tokens.size()[:2]
    beam_size = generator.beam_size

    if constraints is not None and not generator.search.supports_constraints:
        raise NotImplementedError("Target-side constraints were provided, but search method doesn't support them")

    # Initialize constraints, when active
    generator.search.init_constraints(constraints, beam_size)

    max_len: int = -1
    if generator.match_source_len:
        max_len = src_lengths.max().item()
    else:
        max_len = min(
            int(generator.max_len_a * src_len + generator.max_len_b),
            # exclude the EOS marker
            generator.model.max_decoder_positions() - 1,
        )
    assert (
        generator.min_len <= max_len
    ), "min_len cannot be larger than max_len, please adjust these!"
    # compute the encoder output for each beam
    net_input['return_all_hiddens'] = True 
    encoder_outs = generator.model.forward_encoder(net_input)
    encoder_outs_ = generator.model.forward_encoder(net_input)
    # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
    new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
    new_order = new_order.to(src_tokens.device).long()
    encoder_outs = generator.model.reorder_encoder_out(encoder_outs, new_order)
    
    # ensure encoder_outs is a List.
    assert encoder_outs is not None

    # initialize buffers
    scores = (
        torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
    )  # +1 for eos; pad is never chosen for scoring
    tokens = (
        torch.zeros(bsz * beam_size, max_len + 2)
        .to(src_tokens)
        .long()
        .fill_(generator.pad)
    )  # +2 for eos and pad
    tokens[:, 0] = generator.eos if bos_token is None else bos_token
    attn: Optional[Tensor] = None
    
    # A list that indicates candidates that should be ignored.
    # For example, suppose we're sampling and have already finalized 2/5
    # samples. Then cands_to_ignore would mark 2 positions as being ignored,
    # so that we only finalize the remaining 3 samples.
    cands_to_ignore = (
        torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
    )  # forward and backward-compatible False mask

    # list of completed sentences
    finalized = torch.jit.annotate(
        List[List[Dict[str, Tensor]]],
        [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
    )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

    finished = [
        False for i in range(bsz)
    ]  # a boolean array indicating if the sentence at the index is finished or not
    num_remaining_sent = bsz  # number of sentences remaining

    # number of candidate hypos per step
    cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

    # offset arrays for converting between different indexing schemes
    bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
    cand_offsets = torch.arange(0, cand_size).type_as(tokens)

    reorder_state: Optional[Tensor] = None
    batch_idxs: Optional[Tensor] = None
        
    decoder_outs = []
    token_probs = []
    for step in range(max_len + 1):  # one extra step for EOS marker
        
        if reorder_state is not None:
            if batch_idxs is not None:
                # update beam indices to take into account removed sentences
                corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                    batch_idxs
                )
                reorder_state.view(-1, beam_size).add_(
                    corr.unsqueeze(-1) * beam_size
                )
            generator.model.reorder_incremental_state(incremental_states, reorder_state)
            encoder_outs = generator.model.reorder_encoder_out(
                encoder_outs, reorder_state
            )

        decoder_out, lprobs, probs, avg_attn_scores = forward_decoder(
            generator.model,
            tokens[:, : step + 1],
            encoder_outs,
            incremental_states,
            generator.temperature,
        )
        
#         lprobs, avg_attn_scores = generator.model.forward_decoder(
# #             generator.model,
#             tokens[:, : step + 1],
#             encoder_outs,
#             incremental_states,
#             generator.temperature,
#         )

        # decoder の隠れ状態を取得したい場合は forward_decoder をいじり、下をコメントアウト
        decoder_outs.append(decoder_out)
        token_probs.append(probs)
        
        lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

        lprobs[:, generator.pad] = -math.inf  # never select pad
        lprobs[:, generator.unk] -= generator.unk_penalty  # apply unk penalty

        # handle max length constraint
        if step >= max_len:
            lprobs[:, : generator.eos] = -math.inf
            lprobs[:, generator.eos + 1 :] = -math.inf

        # handle prefix tokens (possibly with different lengths)
        if (
            prefix_tokens is not None
            and step < prefix_tokens.size(1)
            and step < max_len
        ):
            lprobs, tokens, scores = generator._prefix_tokens(
                step, lprobs, scores, tokens, prefix_tokens, beam_size
            )
        elif step < generator.min_len:
            # minimum length constraint (does not apply if using prefix_tokens)
            lprobs[:, generator.eos] = -math.inf

        # Record attention scores, only support avg_attn_scores is a Tensor
        if avg_attn_scores is not None:
            if attn is None:
                attn = torch.empty(
                    bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                ).to(scores)
            attn[:, :, step + 1].copy_(avg_attn_scores)

        scores = scores.type_as(lprobs)
        eos_bbsz_idx = torch.empty(0).to(
            tokens
        )  # indices of hypothesis ending with eos (finished sentences)
        eos_scores = torch.empty(0).to(
            scores
        )  # scores of hypothesis ending with eos (finished sentences)

        if generator.should_set_src_lengths:
            generator.search.set_src_lengths(src_lengths)

        if generator.no_repeat_ngram_size > 0:
            lprobs = generator._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)

        # Shape: (batch, cand_size)
        cand_scores, cand_indices, cand_beams = generator.search.step(
            step,
            lprobs.view(bsz, -1, generator.vocab_size),
            scores.view(bsz, beam_size, -1)[:, :, :step],
        )
        
        # cand_bbsz_idx contains beam indices for the top candidate
        # hypotheses, with a range of values: [0, bsz*beam_size),
        # and dimensions: [bsz, cand_size]
        cand_bbsz_idx = cand_beams.add(bbsz_offsets)

        # finalize hypotheses that end in eos
        # Shape of eos_mask: (batch size, beam size)
        eos_mask = cand_indices.eq(generator.eos) & cand_scores.ne(-math.inf)
        eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

        # only consider eos when it's among the top beam_size indices
        # Now we know what beam item(s) to finish
        # Shape: 1d list of absolute-numbered
        eos_bbsz_idx = torch.masked_select(
            cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
        )

        finalized_sents: List[int] = []
        if eos_bbsz_idx.numel() > 0:
            eos_scores = torch.masked_select(
                cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents = generator.finalize_hypos(
                step,
                eos_bbsz_idx,
                eos_scores,
                tokens,
                scores,
                finalized,
                finished,
                beam_size,
                attn,
                src_lengths,
                max_len,
            )
            num_remaining_sent -= len(finalized_sents)

        assert num_remaining_sent >= 0
        if num_remaining_sent == 0:
            break
        assert step < max_len

        # Remove finalized sentences (ones for which {beam_size}
        # finished hypotheses have been generated) from the batch.
        if len(finalized_sents) > 0:
            new_bsz = bsz - len(finalized_sents)

            # construct batch_idxs which holds indices of batches to keep for the next pass
            batch_mask = torch.ones(bsz, dtype=torch.bool, device=cand_indices.device)
            batch_mask[finalized_sents] = False
            # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
            batch_idxs = torch.arange(bsz, device=cand_indices.device).masked_select(batch_mask)

            # Choose the subset of the hypothesized constraints that will continue
            generator.search.prune_sentences(batch_idxs)

            eos_mask = eos_mask[batch_idxs]
            cand_beams = cand_beams[batch_idxs]
            bbsz_offsets.resize_(new_bsz, 1)
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            cand_scores = cand_scores[batch_idxs]
            cand_indices = cand_indices[batch_idxs]

            if prefix_tokens is not None:
                prefix_tokens = prefix_tokens[batch_idxs]
            src_lengths = src_lengths[batch_idxs]
            cands_to_ignore = cands_to_ignore[batch_idxs]

            scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
            tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
            if attn is not None:
                attn = attn.view(bsz, -1)[batch_idxs].view(
                    new_bsz * beam_size, attn.size(1), -1
                )
            bsz = new_bsz
        else:
            batch_idxs = None

        # Set active_mask so that values > cand_size indicate eos hypos
        # and values < cand_size indicate candidate active hypos.
        # After, the min values per row are the top candidate active hypos

        # Rewrite the operator since the element wise or is not supported in torchscript.

        eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
        active_mask = torch.add(
            eos_mask.type_as(cand_offsets) * cand_size,
            cand_offsets[: eos_mask.size(1)],
        )

        # get the top beam_size active hypotheses, which are just
        # the hypos with the smallest values in active_mask.
        # {active_hypos} indicates which {beam_size} hypotheses
        # from the list of {2 * beam_size} candidates were
        # selected. Shapes: (batch size, beam size)
        new_cands_to_ignore, active_hypos = torch.topk(
            active_mask, k=beam_size, dim=1, largest=False
        )

        # update cands_to_ignore to ignore any finalized hypos.
        cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
        # Make sure there is at least one active item for each sentence in the batch.
        assert (~cands_to_ignore).any(dim=1).all()

        # update cands_to_ignore to ignore any finalized hypos

        # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
        # can be selected more than once).
        active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
        active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

        active_bbsz_idx = active_bbsz_idx.view(-1)
        active_scores = active_scores.view(-1)

        # copy tokens and scores for active hypotheses

        # Set the tokens for each beam (can select the same row more than once)
        tokens[:, : step + 1] = torch.index_select(
            tokens[:, : step + 1], dim=0, index=active_bbsz_idx
        )
        # Select the next token for each of them
        tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
            cand_indices, dim=1, index=active_hypos
        )
        if step > 0:
            scores[:, :step] = torch.index_select(
                scores[:, :step], dim=0, index=active_bbsz_idx
            )
        scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
            cand_scores, dim=1, index=active_hypos
        )

        # Update constraints based on which candidates were selected for the next beam
        generator.search.update_constraints(active_hypos)

        # copy attention for active hypotheses
        if attn is not None:
            attn[:, :, : step + 2] = torch.index_select(
                attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
            )

        # reorder incremental state in decoder
        reorder_state = active_bbsz_idx

    # sort by score descending
    for sent in range(len(finalized)):
        scores = torch.tensor([float(elem["score"].item()) for elem in finalized[sent]])
        _, sorted_scores_indices = torch.sort(scores, descending=True)
        finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
        finalized[sent] = torch.jit.annotate(List[Dict[str, Tensor]], finalized[sent])
    return encoder_outs_, decoder_outs, token_probs, finalized


def build_sample(
    tokens, device
):
    reorderd_tokens, ntokens, src_lengths, ids = reodering_tokens(tokens)
    d = {
        'id': ids.to(device),
        'nsentences': len(tokens), 
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': reorderd_tokens.to(device),
            'src_lengths': src_lengths.to(device)
        },
        'target': None # もしかしたら target に参照トークン入れたらティーチャフォースできる？
    }
    return d


def reodering_tokens(tokens):
    token_nums = []
    for token_id in tokens:
        without_mask_token = [x for x in token_id if x != 1]
        token_nums.append(len(without_mask_token))
    
    orders = [x for x in range(len(token_nums))]
    reoders = torch.tensor([
        x for _, x in sorted(zip(token_nums, orders), reverse=True)
    ])
    reodered_token_ids = torch.tensor([
        v for _, v in sorted(zip(token_nums, tokens.tolist()), reverse=True)
    ])
    reodered_token_nums = torch.tensor([sorted(token_nums, reverse=True)])
    
    return reodered_token_ids, sum(token_nums), reodered_token_nums, reoders


def generate(tokens, model, bos_token, beam: int = 1,  **kwargs):
    sample = build_sample(tokens, model.device)
    # sample = model._build_sample(tokens)
    # build dec_generator using current args as well as any kwargs
    gen_args = copy.copy(model.args)
    gen_args.beam = beam
    for k, v in kwargs.items():
        setattr(gen_args, k, v)
    generator = model.task.build_generator([model.model], gen_args)
    encoder_outs, decoder_outs, token_probs, translations = _generate(
        generator,
        sample,
        prefix_tokens=sample['net_input']['src_tokens'].new_zeros((len(tokens), 1)).fill_(model.task.source_dictionary.bos()),
        bos_token=bos_token
    )
    
    def getarg(name, default):
        return getattr(gen_args, name, getattr(model.args, name, default))

    # Process top predictions
    hypos = [x[0] for x in translations]
    hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
    return encoder_outs, decoder_outs, token_probs, hypos, sample['id']

def convert_to_tokens(
    probs, sent_order, d, target_sent_idx
):
    ids = []
    sent_idx = sent_order.tolist().index(target_sent_idx)
    for x in probs:
        try:
            ids.append(torch.argmax(x[sent_idx:sent_idx+1]))
        except:
            break

    return ids_to_tokens(d=d, idxs=ids)