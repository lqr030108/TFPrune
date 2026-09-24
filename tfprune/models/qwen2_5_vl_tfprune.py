"""lmms-eval generation adapter for Qwen2.5-VL."""

import base64
from io import BytesIO
from typing import List

import torch
from accelerate import Accelerator
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from tfprune.qwen import tfprune_qwen2_5_vl

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:
    raise ImportError("Install the Qwen environment described in README.md") from exc


@register_model("qwen2_5_vl_tfprune")
class Qwen2_5_VL_TFPrune(lmms):
    def __init__(
        self,
        pretrained="Qwen/Qwen2.5-VL-7B-Instruct",
        target_retention_ratio=0.222,
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
        if accelerator.num_processes != 1:
            raise ValueError("Use a single process; device_map=auto supports model sharding")
        self._device = torch.device(device)
        self.device_map = device_map
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        self.min_pixels = 200704
        self.max_pixels = 1605632
        self.processor = AutoProcessor.from_pretrained(
            pretrained, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
        self.system_prompt = "You are a helpful assistant."
        self._config = self._model.config
        self._max_length = 2048
        self.batch_size_per_gpu = 1
        self.use_cache = True
        self._rank = 0
        self._world_size = 1
        self._model, self.processor = tfprune_qwen2_5_vl(
            self._model, self.processor, target_retention_ratio=float(target_retention_ratio)
        )

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

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return (-len(toks), x[0])

        pbar = tqdm(total=len(requests), disable=self.rank != 0, desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            if len(visual_list[0] or []) != 1 or not isinstance(visual_list[0][0], Image.Image):
                raise ValueError("This adapter requires exactly one PIL image per sample")
            gen_kwargs = dict(all_gen_kwargs[0])
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(
                    f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}"
                )
            until = [item for item in until if item != "\n\n"]
            if isinstance(contexts, tuple):
                contexts = list(contexts)
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")
            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")
                message = [{"role": "system", "content": self.system_prompt}]
                processed_visuals = []
                for visual in visual_list[i] or []:
                    if isinstance(visual, Image.Image):
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        processed_visuals.append(
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_string}",
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )
                message.append(
                    {
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    }
                )
                batched_messages.append(message)
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batched_messages
            ]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                raise NotImplementedError("This release supports image tasks only")
            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            if current_gen_kwargs["num_beams"] != 1:
                raise ValueError("Qwen2.5-VL TFPrune currently supports num_beams=1 only")
            pad_token_id = self.tokenizer.pad_token_id
            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans
            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Only single-turn generation is supported")

    def loglikelihood(self, requests):
        raise NotImplementedError("Only generation tasks are supported")
