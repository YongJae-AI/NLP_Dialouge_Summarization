from typing import Tuple

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .utils import get_dialogue_special_tokens


def load_t5_model_and_tokenizer(
    model_name: str,
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # T5/mT5 등 다른 백본에서는,
    # 해당 문자열이 현재 tokenizer에서 전부 UNK로만 인코딩되는 경우에만
    # 진짜 신규 special token으로 추가한다.
    specials = get_dialogue_special_tokens()
    unk_id = tokenizer.unk_token_id
    tokens_to_add = []
    for tok in specials:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        # 이미 subword 조합으로 잘 표현되면(ids 중에 unk가 없으면) 추가하지 않는다.
        if ids and all(i != unk_id for i in ids):
            continue
        tokens_to_add.append(tok)

    if tokens_to_add:
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens_to_add},
        )
        if added > 0:
            model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
