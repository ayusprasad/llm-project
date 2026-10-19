from app import build_messages


def test_build_messages_includes_system_prompt_and_current_question():
    messages = build_messages(" Be helpful ", [], " What is AI? ", memory_turns=3)

    assert messages == [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "What is AI?"},
    ]


def test_build_messages_limits_history_to_requested_turns():
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": "unfinished"},
    ]

    messages = build_messages("", history, "next", memory_turns=2)

    assert messages[0]["content"] == "You are a helpful, concise assistant."
    assert messages[1:] == [
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": "unfinished"},
        {"role": "user", "content": "next"},
    ]


def test_build_messages_with_zero_memory_keeps_only_current_turn():
    messages = build_messages(
        "system",
        [{"role": "user", "content": "prior"}, {"role": "assistant", "content": "answer"}],
        "now",
        0,
    )

    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "now"},
    ]
