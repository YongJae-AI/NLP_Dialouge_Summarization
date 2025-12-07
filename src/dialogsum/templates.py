"""
Structured encoder/decoder(CoT) templates for experimentation.
"""

encoder_templates = {
    "enc_guide_block": """[요약 지침]
- 대화의 핵심 내용만 정리하라.
- 모든 화자 역할(#PersonX#)을 원문 그대로 유지하라.
- 새로운 정보를 추가하지 말라.
- 사건 순서대로 간결하게 요약하라.

[대화]
{dialogue}
""",
    "enc_role_preserve": """아래 대화를 요약하시오.
화자 역할(#PersonX#)은 절대 변경하지 마시오.
대화의 사실만 남겨 간결하게 정리하시오.

대화:
{dialogue}
""",
    "enc_3step_extract": """아래 대화를 요약하기 위해 다음 정보를 추출하시오:
1) 주요 화자와 역할
2) 핵심 사건
3) 결론

대화:
{dialogue}
""",
}

cot_templates = {
    "cot_standard_reasoning": """<think>
1) 대화의 주요 주제가 요약문에 포함하도록 하라.
2) 사건이 발생한 순서를 정리하라.
3) 핵심 정보만 요약문에 포함하도록 하라.
4) 화자 역할(#PersonX#)이 유지되었는지 확인하라.
</think>

최종 요약:""",
    "cot_dialogue_decomp": """<analysis>
- 각 화자의 목적/의도를 정리하라.
- 상호작용 중 발생한 핵심 사건을 나열하라.
- 결론 또는 상태 변화가 무엇인지 정리하라.
</analysis>

<summary>""",
    "cot_hierarchical": """<think>
1) 대화를 세 부분으로 나누어 각 부분의 요지를 요약하라.
2) 세 요약을 기반으로 전체 요약을 통합하라.
</think>

최종 요약:""",
    "cot_reflexion": """<think>
1) 요약이 원문의 20% 이내인지 확인하라.
2) 대화에 없는 정보를 넣지 않았는지 점검하라.
3) 화자 역할(#PersonX#)이 바뀌지 않았는지 확인하라.
4) 논리적으로 자연스러운지 검토하라.
</think>

수정된 최종 요약:""",
    "cot_light_planning": """<think>
- 대화의 목적은 무엇인가?
- 가장 중요한 두세 가지 사건은 무엇인가?
- 최종 상태 또는 결론은 무엇인가?
</think>

요약:""",
}
