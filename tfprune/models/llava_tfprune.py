"""lmms-eval generation adapter for LLaVA."""

from typing import List

import torch
from accelerate import Accelerator
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from loguru import logger as eval_logger
from tqdm import tqdm

from tfprune import tfprune

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
except ImportError as exc:
    raise ImportError("Install the LLaVA environment described in README.md") from exc


@register_model("llava_tfprune")
class Llava_TFprune(lmms):
    def __init__(
        self,
        pretrained="liuhaotian/llava-v1.5-7b",
        target_vision_tokens=128,
        device="cuda:0",
        device_map="auto",
        batch_size=1,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(f"Unsupported model arguments: {sorted(kwargs)}")
        if int(batch_size) != 1:
            raise ValueError("TFPrune requires batch_size=1")
        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes != 1:
            raise ValueError("Use a single process; device_map=auto supports model sharding")
        self._device = torch.device(device)
        self.device_map = device_map
        name = get_model_name_from_path(pretrained)
        self._tokenizer, self._model, self._image_processor, self._max_length = (
            load_pretrained_model(
                pretrained,
                None,
                name,
                device_map=device_map,
                attn_implementation="eager",
                multimodal=True,
            )
        )
        self._model.eval()
        self._model.tie_weights()
        self._config = self._model.config
        self.truncation = True
        self.batch_size_per_gpu = 1
        self.conv_template = "vicuna_v1"
        self.use_cache = True
        self._rank = 0
        self._world_size = 1
        if not getattr(self._tokenizer, "is_fast", False):
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
        self._model = tfprune(self._model, target_vision_tokens=int(target_vision_tokens))

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def flatten(self, input):
        if not input or any((i is None for i in input)):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return (-len(toks), x[0])

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=self.rank != 0, desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            flattened_visuals = self.flatten(batched_visuals)
            if len(flattened_visuals) != 1:
                raise ValueError("This adapter requires exactly one image per sample")
            gen_kwargs = dict(all_gen_kwargs[0])
            until = [self.tok_decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}"
                    )
            if (
                "image_aspect_ratio" in gen_kwargs.keys()
                and "image_aspect_ratio" not in self._config.__dict__
            ):
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
            if flattened_visuals:
                image_tensor = process_images(
                    flattened_visuals, self._image_processor, self._config
                )
                if type(image_tensor) is list:
                    image_tensor = [
                        _image.to(dtype=torch.float16, device=self.device)
                        for _image in image_tensor
                    ]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            else:
                image_tensor = None
            question_input = []
            for visual, context in zip(batched_visuals, contexts):
                if (
                    image_tensor is not None
                    and len(image_tensor) != 0
                    and (DEFAULT_IMAGE_TOKEN not in context)
                ):
                    image_tokens = (
                        [DEFAULT_IMAGE_TOKEN] * len(visual)
                        if isinstance(visual, list)
                        else [DEFAULT_IMAGE_TOKEN]
                    )
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context
                conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)
            gen_kwargs["image_sizes"] = [
                flattened_visuals[idx].size for idx in range(len(flattened_visuals))
            ]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if gen_kwargs["num_beams"] != 1:
                raise ValueError("TFPrune supports num_beams=1 only")
            input_ids_list = [
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                for prompt in question_input
            ]
            pad_token_ids = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
            input_ids = self.pad_sequence(
                input_ids_list, batch_first=True, padding_value=pad_token_ids
            ).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            cont = self.model.generate(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                images=image_tensor,
                image_sizes=gen_kwargs["image_sizes"],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            for i, answer in enumerate(text_outputs):
                for term in until:
                    if term:
                        answer = answer.split(term)[0]
                text_outputs[i] = answer
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Only single-turn generation is supported")

    def loglikelihood(self, requests):
        raise NotImplementedError("Only generation tasks are supported")
