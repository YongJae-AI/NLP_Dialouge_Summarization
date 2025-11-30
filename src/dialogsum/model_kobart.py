from typing import Tuple

from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast


def load_kobart_model_and_tokenizer(
    model_name: str,
) -> Tuple[BartForConditionalGeneration, PreTrainedTokenizerFast]:
    """
    KoBART용 로더.
    주어진 대회용 특수 토큰(#Person1#, #SSN# 등)은
    이미 사전 학습 시점의 vocab/subword 조합으로 잘 표현되므로
    별도의 add_special_tokens 호출을 하지 않는다.
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer
