# !/opt/conda/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Model, PretrainedConfig, AutoConfig, MODEL_MAPPING
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
import sys
from transformers.utils import logging
from transformers.file_utils import WEIGHTS_NAME
from transformers.modeling_utils import unwrap_model
import os
from utils import get_init_prefix_weight

logger = logging.get_logger(__name__)

sys.path.append("..")

class PrefixGPT2Config(PretrainedConfig):
    model_type = "prefix_gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
            self,
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            gradient_checkpointing=False,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            base_model_name='gpt2',
            num_token_of_prefix=128,
            init_prefix='random',
            **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, base_model_name=base_model_name,
                         num_token_of_prefix=num_token_of_prefix, init_prefix=init_prefix, **kwargs)

        self.vocab_size = vocab_size
        self.n_ctx = n_ctx #Dimensionality of the causal mask 
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner  #Dimensionality of the inner feed-forward layers
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.gradient_checkpointing = gradient_checkpointing
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer


class PrefixGPT2LMHeadModel(GPT2PreTrainedModel):
    config_class = PrefixGPT2Config
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight", 'transformer']
    base_model_prefix = 'prefix_embed'
    def __init__(self, config):
        super().__init__(config)
        self.num_token_of_prefix = config.num_token_of_prefix
        self.base_model_name = config.base_model_name
        base_model = config.base_model_name
        print("[Loading PLMs from {}]".format(base_model))
        self.transformer = GPT2Model.from_pretrained(base_model)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tie_weights() #Tie the weights between the input embeddings and the output embeddings
        self.prefix_embed = nn.Embedding(config.num_token_of_prefix, config.n_embd)
        self.freeze_non_prefix()
        self.prefix = torch.arange(0, config.num_token_of_prefix).long().unsqueeze(0) 
        self.model_parallel = False
        self.device_map = None
        self._prefix_to_save = 'prefix_embed'

    def init_prefix(self, init_prefix):
        weight = get_init_prefix_weight(init_prefix, self.base_model_name, self.get_input_embeddings(),
                                        self.num_token_of_prefix)
        self.prefix_embed.weight.data.copy_(weight)

    def freeze_non_prefix(self):
        # freeze parameters of pre-trained models
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False
        for param in self.prefix_embed.parameters():
            param.requires_grad = True

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        # attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        num_token_of_prefix = self.num_token_of_prefix
        attention_mask = torch.ones(input_ids.size(0), num_token_of_prefix + input_ids.size(1)).long().to(
            input_ids.device)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "do_generate": True,
        }

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            topic=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            do_generate=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """

        if do_generate:  # generate
            if past_key_values is None:  # create prefix in the first step in generation
                prefix = self.prefix.repeat(input_ids.size(0), 1).to(input_ids.device)
                prefix_embeds = self.prefix_embed(prefix)
                inputs_embeds = self.transformer.get_input_embeddings()(input_ids)
                # B*L*D, cat(prefix, input)
                inputs_embeds = torch.cat((prefix_embeds, inputs_embeds), dim=1)
                input_ids = None
            else:
                assert input_ids is not None
                inputs_embeds = None
        else:  # train
            assert inputs_embeds is None
            prefix = self.prefix.repeat(input_ids.size(0), 1).to(input_ids.device)
            # origin [batch_size, seq_len]

            attention_mask = torch.cat((torch.ones(input_ids.size(0), self.num_token_of_prefix).long().to(input_ids.device), attention_mask), dim=1)
            
            if labels is not None:
                labels = torch.cat((torch.ones(input_ids.size(0), self.num_token_of_prefix).fill_(-100).long().to(
                    input_ids.device), labels), dim=1)
            prefix_embeds = self.prefix_embed(prefix)
            # prefix_embeds = self.prefix_dropout(prefix_embeds)
            inputs_embeds = self.transformer.get_input_embeddings()(input_ids)
            # B*L*D, cat(prefix, input)


            inputs_embeds = torch.cat((prefix_embeds, inputs_embeds), dim=1)

            input_ids = None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output


        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

    def save_pretrained(self, save_directory, save_config = True, state_dict = None, save_function = torch.save):
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        os.makedirs(save_directory, exist_ok=True)
        model_to_save = unwrap_model(self)
        model_to_save.config.architectures = [model_to_save.__class__.__name__]
        if save_config:
            model_to_save.config.save_pretrained(save_directory)

        if state_dict is None:
            state_dict = model_to_save.state_dict()
        if self._keys_to_ignore_on_save is not None:
            state_dict = {k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save}
        if self._prefix_to_save is not None:
            state_dict = {k: v for k, v in state_dict.items() if k.startswith(self._prefix_to_save)}
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(state_dict, output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))
