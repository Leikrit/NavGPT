from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers import pipeline

class Custom_Model(LLM):
    pipe: Any  #: :meta private:

    """Key word arguments passed to the model."""
    temperature: float = 0.6
    top_p: float = 0.9
    max_seq_len: int = 128
    max_gen_len: int = 64
    max_batch_size: int = 4

    @property
    def _llm_type(self) -> str:
        return "custom_model"

    @classmethod
    def from_model_id(
        cls,
        model_name: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 128,
        max_gen_len: int = 64,
        max_batch_size: int = 4,
        **kwargs: Any,
    ) -> LLM:
        """Construct the pipeline object from model_id and task."""
        pipe  = pipeline("text-generation", model_name, torch_dtype="auto", device_map="auto")
        return cls(
            pipe = pipe,
            # set as default
            temperature = 0.6,
            top_p = top_p,
            max_seq_len = max_seq_len,
            max_gen_len = max_gen_len,
            max_batch_size = max_batch_size,
            **kwargs,
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.pipe(messages, max_new_tokens=self.max_gen_len)[0]["generated_text"][-1]["content"]
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_seq_len": self.max_seq_len,
            "max_gen_len": self.max_gen_len,
            "max_batch_size": self.max_batch_size,
            }
