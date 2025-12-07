#!/usr/bin/env python
# 간단한 테스트: style_prompt만 사용했을 때와 encoder_template를 사용했을 때의 최종 encoder 입력을 출력

def show_style_prompt(style_prompt: str, dialogue: str) -> None:
    print("=== style_prompt 결과 ===")
    final = f"{style_prompt}\n\n대화:\n{dialogue}\n\n답변:"
    print(final)
    print()


def show_encoder_template(template: str, dialogue: str) -> None:
    print("=== encoder_template 결과 ===")
    final = template.format(dialogue=dialogue)
    print(final)
    print()


def main() -> None:
    style_prompt = "질문: 아래 한국어 대화를 요약하시오. 답변:"
    encoder_template = """[요약 지침]
- 대화의 핵심 내용만 정리하라.
- 모든 화자 역할(#PersonX#)을 원문 그대로 유지하라.
- 새로운 정보를 추가하지 말라.
- 사건 순서대로 간결하게 요약하라.

[대화]
{dialogue}
"""
    dialogue = "안녕? 오늘 날씨 어때?"

    show_style_prompt(style_prompt, dialogue)
    show_encoder_template(encoder_template, dialogue)


if __name__ == "__main__":
    main()
