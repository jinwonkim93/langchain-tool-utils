# langchain-tool-utils

LangChain agent의 메시지를 정리하는 함수입니다. **정리된 메시지 목록만 반환**하며, 입력을 변경하거나 도구를 실행하지 않습니다.

```python
from langchain_tool_utils import clean_messages

messages = clean_messages(messages)  # API와 독립적인 메시지 정리
```

## 처리 규칙

- `message.id`가 중복이면 마지막 메시지를 유지합니다. ID가 없는 메시지는 서로 다른 메시지로 취급합니다.
- `tool_call.id`가 중복이면 마지막 호출을 유지합니다. 이름·인자를 추측하거나 호출을 분리하지 않습니다.
- 같은 `tool_call_id`의 결과는 실제 결과를 합성 `PENDING`보다 우선합니다. 그 외에는 마지막 결과를 유지합니다.
- 전체 이력에 대응하는 AI 호출이 없는 ToolMessage는 삭제합니다.
- 결과는 대응하는 AI 바로 뒤로 이동합니다. 중간에 외부 Human/AI가 들어왔거나 결과가 늦게 도착한 경우도 처리합니다.
- 결과가 없는 호출에는 동일한 call ID의 `PENDING` ToolMessage를 보충합니다.
- 실제 도구 오류의 ToolMessage는 오류 상태와 내용을 유지합니다.
- 미완성 `AIMessageChunk` 등 stream chunk는 저장할 메시지로 승격하지 않고 제거합니다.
- 긴 ID나 허용하지 않는 문자가 들어간 call ID는 호출·결과를 함께 안정적으로 변환합니다. 기본 정책은 ASCII 영숫자·`_`·`-`, 최대 40자입니다. 이는 프로젝트의 보수적 정책이며 모든 provider의 공식 제한이 같다는 의미는 아닙니다. `max_call_id_length`로 조절할 수 있습니다.
- 정리 후 전체 메시지 구조와 호출별 결과가 정확히 하나인지 검사합니다. API 직렬화 검증은 `validate_messages(messages, api=...)`로 분리했습니다. 기존 `clean_messages(..., api=...)`도 선택적 추가 검증으로 지원합니다.

`clean_messages(clean_messages(messages)) == clean_messages(messages)`가 성립합니다. 잘못된 JSON 인자를 추측하거나 성공 결과를 만들어내지 않습니다. 완성된 AIMessage의 깨진 인자, 지원하지 않는 content block 등은 `UnsafeHistoryError`를 발생시켜 잘못된 요청을 차단합니다.

## PENDING과 ERROR

합성 결과의 `content`는 다음 형태입니다.

```json
{
  "state": "PENDING",
  "reason": "result_missing",
  "message": "No final result is available. Check whether the operation is still running or already took effect. Request the tool again when retrying is safe."
}
```

LangChain의 ToolMessage.status는 `success`/`error`만 지원하므로 합성 결과는 `status="error"`를 쓰고, 서비스 의미는 `content.state="PENDING"`으로 표현합니다. 내부 표식은 `additional_kwargs`에 들어 있어 실제 결과와 구분됩니다. 실제 결과가 뒤늦게 들어오면 다시 정리할 때 PENDING이 제거됩니다. ID를 정규화한 경우 늦게 도착하는 원래 call ID도 인식하도록 매핑을 AIMessage 내부에 보존합니다. 별도의 보고서나 로그를 반환하지 않습니다.

PENDING은 실행 여부가 불명확하다는 뜻입니다. AI가 재호출을 판단할 수 있지만, 결제·메일 같은 도구의 중복 실행 방지는 도구의 권한 검사와 idempotency 정책에서 처리해야 합니다. 재호출은 새로운 call ID를 사용합니다.

## 검사만 하기

```python
from langchain_tool_utils import validate_messages

validate_messages(messages, api="responses")  # 정상: None, 비정상: 예외
```

검사에는 API key나 네트워크가 필요하지 않습니다. 실제 `langchain-openai` 직렬화 함수를 사용하며 의존 버전을 고정된 범위로 제한합니다. 지원 범위는 로컬 function tool, 텍스트, Human image_url, Responses reasoning, refusal 블록입니다. 모델의 기능 지원, 토큰 한도, tool schema, 인증과 네트워크까지 검증하는 함수는 아닙니다. 비어 있는 결과 목록은 API 요청 전에 새 사용자/시스템 입력 등 애플리케이션 문맥과 함께 사용하세요.

## 정리한 메시지를 state에 저장

일반 dict는 그대로 교체합니다.

```python
state["messages"] = clean_messages(state["messages"])
```

LangGraph의 `add_messages` reducer에서는 목록을 반환하면 append/merge되므로, 삭제까지 반영하려면 명시적으로 교체합니다.

```python
from langchain.agents.middleware import AgentMiddleware
from langgraph.types import Overwrite
from langchain_tool_utils import clean_messages

class CleanMessagesMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        cleaned = clean_messages(state["messages"])
        if cleaned == state["messages"]:
            return None
        return {"messages": Overwrite(value=cleaned)}

    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)
```

정리 결과는 checkpointer에 저장됩니다. 이후 호출에서는 동일한 항목을 다시 만들거나 변경하지 않습니다. 새 입력이 들어왔는지 확인하기 위한 선형 검사는 수행합니다. 같은 thread의 입력 주입·정리·저장은 직렬화하거나 저장소 버전 검사로 보호해야 합니다. 이미 reducer가 기존 메시지를 덮어쓴 뒤에는 잃어버린 내용을 복구할 수 없으므로, 외부 메시지 병합에 중복 우선순위를 적용하려면 저장된 목록과 새 목록을 합친 뒤 정리하고 교체하세요.

## create_agent에서 API 전환

```python
import os
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_tool_utils import create_healing_agent

@tool
def lookup(order_id: str) -> str:
    """주문 상태 조회."""
    return f"{order_id}: processing"

with SqliteSaver.from_conn_string("agent.sqlite") as saver:
    agent = create_healing_agent(
        model=os.environ["OPENAI_MODEL"],  # 두 API를 지원하는 모델
        tools=[lookup],
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": "order-123", "api": "responses"}}
    agent.invoke({"request_messages": [("user", "주문 123 확인")]}, config, durability="sync")
    config["configurable"]["api"] = "chat_completions"
    agent.invoke({"request_messages": [("user", "다음 단계는?")]}, config, durability="sync")
```

`OPENAI_API_KEY`가 필요합니다. `clean_messages`는 agent나 DB 없이 사용할 수 있습니다. middleware는 누락된 결과를 PENDING으로 정리하고 도구 실행 예외를 ERROR로 전달합니다. 도구의 실제 실행 여부를 추적하거나 중복 실행을 막지는 않습니다.

같은 `thread_id`에서 API가 바뀌면 저장된 messages를 초기화하고 **이번 호출의 새 메시지만** 남깁니다. 같은 API에서는 이력을 유지합니다. API를 다시 바꿔도 이전 이력을 복원하지 않습니다. reasoning을 포함한 이전 API 전용 데이터는 새 API에 전달하지 않습니다.

factory는 `create_agent`의 **CompiledStateGraph를 그대로 반환**합니다. 새 요청은 `messages` 대신 `request_messages`로 전달하세요. 기존 `ApiSessionAgent` 래퍼는 제거했습니다.

역할은 다음처럼 분리합니다.

- `ApiSessionMiddleware.before_agent`: 저장된 `session_api`와 요청 API를 비교합니다. 같으면 기존 이력 + 요청 메시지, 다르면 요청 메시지만 선택합니다. `Overwrite`로 저장하고 `request_messages`를 비웁니다. 메시지를 정리하거나 결과를 생성하지 않습니다.
- `ToolHealingMiddleware.before_agent` / `before_model`: 선택된 메시지를 `clean_messages`로 정리하고, 달라졌을 때만 `Overwrite`로 저장합니다.
- `clean_messages`: API 변경이나 세션 초기화를 판단하지 않습니다. 전달받은 메시지 관계만 정리합니다.

중단된 실행은 원래 API로 재개·완료한 뒤 전환해야 합니다. 재개가 `before_agent`를 건너뛰더라도 모델·도구 실행 직전 검사로 API 변경을 차단합니다. API 정보가 없는 기존 checkpoint는 새 `request_messages`가 들어오면 해당 목록으로 초기화합니다. `session_api`, `session_run_active`는 middleware 내부 필드이므로 호출자가 설정하지 마세요.

일반 graph이므로 middleware 검증 전에 입력 자체가 checkpoint에 저장될 수 있습니다. 거부된 요청을 재시도할 때는 올바른 API와 `request_messages`를 명시하세요. 같은 thread의 실행·외부 state 수정은 서버에서 직렬화해야 합니다. 비메시지 state와 이전 checkpoint snapshot은 초기화하지 않습니다. 시스템 지침은 `system_prompt`로 설정하면 전환 후에도 적용됩니다.

`portable_history`는 명시적으로 요청할 때만 쓰는 기존 호환 함수이며 자동 전환 경로에서는 사용하지 않습니다. provider 내장 tool과 legacy function_call은 지원하지 않습니다. `heal_history`는 메시지 목록만 반환합니다.

## FastAPI / SSE 중단

클라이언트 SSE 종료와 도구 실패는 다른 상태입니다. 연결이 끊겼다고 실행 중인 도구를 실패 처리하거나 재실행하지 마세요. 같은 run에 재접속하거나 실행 종료를 확인한 뒤 같은 checkpoint를 재개합니다.

`watch_stream(agent.astream(...), idle_timeout=90, total_timeout=300)`은 이벤트가 멈춘 async stream을 취소합니다. 부분 chunk를 state에 AIMessage로 넣지 않습니다. `create_healing_agent`는 모델의 정상 종료 이벤트가 없는 tool call을 실행하지 않습니다. HTTP read timeout도 설정됩니다.

graph 이벤트가 없는 긴 도구 실행은 watchdog에 걸릴 수 있고, 동기 작업이나 취소를 무시하는 소스를 강제 종료할 수는 없습니다. `agent.invoke(None, config)`나 `Command(resume=...)` 재개는 LangGraph의 기본 동작을 따르며 도구가 다시 실행될 수 있습니다. 실제 효과 확인과 재시도 정책은 애플리케이션 책임입니다. 새 요청으로 실행하면 `before_agent`에서 기존 이력의 누락 결과를 PENDING으로 정리합니다.

## 설치와 테스트

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests examples
```

고정된 테스트 의존성은 `requirements-test.lock`에 있습니다. 테스트는 API key 없이 실제 LangChain 직렬화와 모의 HTTP/SSE를 검증합니다. 실제 하위 프로세스를 종료한 뒤 디스크 checkpoint를 읽어 새 요청에서 누락된 결과를 정리하는 테스트도 포함합니다.

추가 조합 테스트는 짧은 이력 4,681개, 7개 이벤트 순열 5,040개, 고정 seed의 긴 이력 100개를 독립적인 기대값 계산과 비교합니다. JSON 저장·복원 전후 멱등성, 입력 불변성, 두 API 직렬화, 늦은 결과와 ID 길이 변경도 검사합니다. 유한 범위를 전수 검사하며 모든 가능한 입력이나 실제 API 무오류를 보장하는 것은 아닙니다.

서빙 계층에서는 같은 thread의 실행과 checkpoint 교체를 직렬화해야 합니다. 이 모듈은 동시 graph 실행의 전체 state 저장을 잠그지 않습니다.

[설계와 참고 자료](docs/design.md)
