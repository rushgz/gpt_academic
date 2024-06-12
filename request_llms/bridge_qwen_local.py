model_name = "Qwen_Local"
cmd_to_install = "`pip install -r request_llms/requirements_qwen_local.txt`"

from toolbox import ProxyNetworkActivate, get_conf
from .local_llm_class import LocalLLMHandle, get_local_llm_predict_fns


# ------------------------------------------------------------------------------------------------------------------------
# 🔌💻 Local Model
# ------------------------------------------------------------------------------------------------------------------------
class GetQwenLMHandle(LocalLLMHandle):

    def chat(self, model, tokenizer, query, history=[], system_prompt="You are a helpful assistant."):
        conversation = [
            {'role': 'system', 'content': system_prompt},
        ]
        for query_h, response_h in history:
            conversation.append({'role': 'user', 'content': query_h})
            conversation.append({'role': 'assistant', 'content': response_h})
        conversation.append({'role': 'user', 'content': query})
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def chat_stream(self, model, streamer, tokenizer, query, history=[], system_prompt="You are a helpful assistant."):
        conversation = [
            {'role': 'system', 'content': system_prompt},
        ]
        for query_h, response_h in history:
            conversation.append({'role': 'user', 'content': query_h})
            conversation.append({'role': 'assistant', 'content': response_h})
        conversation.append({'role': 'user', 'content': query})
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        from threading import Thread
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)

        thread.start()
        for new_text in streamer:
            yield new_text

    def load_model_info(self):
        # 🏃‍♂️🏃‍♂️🏃‍♂️ 子进程执行
        self.model_name = model_name
        self.cmd_to_install = cmd_to_install

    def load_model_and_tokenizer(self):
        # 🏃‍♂️🏃‍♂️🏃‍♂️ 子进程执行
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        if self._model:
            pass
        with ProxyNetworkActivate('Download_LLM'):
            model_id = get_conf('QWEN_LOCAL_MODEL_SELECTION')
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, resume_download=True)
            # use fp16
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # model.generation_config = GenerationConfig.from_pretrained(model_id, trust_remote_code=True)  # 可指定不同的生成长度、top_p等相关超参
            self._model = model
            self._streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


        return self._model, self._tokenizer

    def llm_stream_generator(self, **kwargs):
        # 🏃‍♂️🏃‍♂️🏃‍♂️ 子进程执行
        def adaptor(kwargs):
            query = kwargs['query']
            max_length = kwargs['max_length']
            top_p = kwargs['top_p']
            temperature = kwargs['temperature']
            history = kwargs['history']
            return query, max_length, top_p, temperature, history

        query, max_length, top_p, temperature, history = adaptor(kwargs)

        responses = ""
        for response in self.chat_stream(self._model, self._streamer, self._tokenizer, query, history=history):
            responses += response
            yield responses

    def try_to_import_special_deps(self, **kwargs):
        # import something that will raise error if the user does not install requirement_*.txt
        # 🏃‍♂️🏃‍♂️🏃‍♂️ 主进程执行
        import importlib
        importlib.import_module('modelscope')


# ------------------------------------------------------------------------------------------------------------------------
# 🔌💻 GPT-Academic Interface
# ------------------------------------------------------------------------------------------------------------------------
predict_no_ui_long_connection, predict = get_local_llm_predict_fns(GetQwenLMHandle, model_name)