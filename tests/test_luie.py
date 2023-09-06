from luie import LuieEngine


def test_run_luie_engine_ner():
    engine = LuieEngine(task="ner")

    print(engine.run("홍길동의 아버지는 홍판서이다."))
