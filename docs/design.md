# 설계와 참고 자료

2026-10-26 확인. 아래 공개 자료의 복구 원칙을 참고해 LangChain middleware로 별도 구현했습니다. 다른 프로젝트의 기능 전체를 이식하거나 동일한 안전성을 주장하는 구현은 아닙니다.

## 참고한 동작

1. [OpenCode v1.2.24 processor.ts](https://github.com/anomalyco/opencode/blob/v1.2.24/packages/opencode/src/session/processor.ts): 프로세서 정리 단계에서 완료/오류 상태가 아닌 tool part를 오류 상태로 닫습니다. 이 프로젝트도 미완료를 성공 처리하지 않는 원칙을 사용합니다.
2. [Cline 공식 저장소의 architecture guide](https://github.com/cline/cline/blob/main/.clinerules/cline-overview.md#task-state--resumption): 재개 시 중단된 tool use에 대응하는 tool result를 보충하고 작업 상태를 다시 평가합니다. 이 프로젝트는 마지막 메시지뿐 아니라 전체 call group을 검사해 외부 Human/AI 주입으로 중간에 끊긴 그룹도 복구합니다. 참고한 자료는 저장소의 설명 문서이며 현재 버전 실행 코드를 전부 검증한 것은 아닙니다.
3. [Claude Code session 문서](https://code.claude.com/docs/en/sessions#resume-a-session): 재개 시 중단된 도구의 결과가 기록되지 않았음을 알리고 실제 효과 확인 후 다음 작업을 정하는 동작을 설명합니다. 이 프로젝트의 메시지 정리 함수는 결과 불명 상태를 PENDING으로 표시하며 도구를 실행하지 않습니다. Claude Code 내부 구현은 공개 문서 이상의 추정을 하지 않았습니다.
4. [LangChain middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom), [wrap_tool_call API](https://reference.langchain.com/python/langchain/agents/middleware/types/AgentMiddleware/wrap_tool_call): before_agent/before_model에서 canonical state를 정리하고 wrap_model_call/wrap_tool_call에서 프로토콜과 실행을 분리합니다. sync/async 경로를 모두 구현했습니다.
5. [ChatOpenAI integration](https://docs.langchain.com/oss/python/integrations/chat/openai): `use_responses_api`를 명시하고 두 모델 인스턴스를 같은 graph에서 선택합니다. 서버 continuation을 사용하지 않습니다. 같은 API의 이력만 전달하고 API 전환 시 messages를 초기화합니다.

## 메시지 정리와 실행

정리 함수는 실제 도구 실행 상태를 조회하지 않습니다. AI 호출에 결과가 없으면
PENDING 결과를 보충하고, 실제 도구 예외는 middleware에서 ERROR로 변환합니다.
도구 재시도, 중복 실행 방지, 외부 state 주입의 실행 권한은 애플리케이션 책임입니다.
`invoke(None)`과 `Command(resume=...)`은 LangGraph의 기본 재개 의미를 따릅니다.

모델 응답은 모든 chunk 수신 후 정상 종료를 검증합니다. Chat Completions는 tool 호출의 `finish_reason=tool_calls` 또는 최종 텍스트의 `stop`, Responses는 `status=completed`를 확인합니다. 인자가 JSON으로 파싱되더라도 조기 EOF, `length`, `incomplete`, 미완료 chunk는 실행하지 않습니다. SSE timeout 시 부분 출력을 완성된 AIMessage로 승격하지 않습니다.

독립 함수 `clean_messages`는 DB 없이 정리된 목록만 반환합니다. 중복 메시지/호출은 마지막 항목을 유지하고, 실제 결과는 합성 PENDING보다 우선합니다. 고아 결과는 삭제하고, 뒤늦은 결과도 전체 이력의 call ID로 연결합니다. 검사 전용 함수 `validate_messages`는 성공하면 None, 실패하면 예외를 반환합니다. 별도 repair 보고서는 없습니다.

정리된 목록은 checkpoint에 저장합니다. 세션 API가 바뀌면 before_agent에서 별도 request_messages를 사용해 과거 messages를 초기화합니다. 같은 API에서는 reasoning을 포함한 이력을 보존합니다. 같은 thread의 동시 쓰기를 직렬화하는 것은 호스트 책임입니다.



## 테스트

테스트는 외부 API 키 없이 실행됩니다. fake model만으로 끝내지 않고 실제 `ChatOpenAI` SDK/직렬화/파서를 HTTP MockTransport에 연결해 두 API의 요청 형식과 SSE를 검증합니다. 실제 OpenAI 서버나 모델 판단 품질을 검증하는 live 테스트는 아닙니다.

- 실제 create_agent의 정상/예외/인자 검증/알 수 없는 도구/병렬 도구
- 양방향 API 전환 시 기존 function call/result 제거, 같은 API에서는 이력 유지
- 두 API의 정상 SSE, terminal event 없는 조기 EOF, 완성 JSON의 truncated 응답
- stalled SSE transport 종료 및 미완료 model state 미커밋
- idle timeout, heartbeat를 포함한 total deadline, async 취소
- 누락/중복/충돌/고아 tool result, 외부 Human/AI 주입, 외부 메시지 입력
- tools 이전 checkpoint interrupt와 재개, 도구 본문 interrupt의 기본 재개
- ToolMessage와 Command 결과의 ID 검증 및 state 업데이트
- 실제 하위 프로세스 `os._exit`로 실행 중 종료 후, 디스크 checkpoint를 다시 열어 새 요청에서 누락된 결과를 PENDING으로 정리

검증 환경: Python 3.14.2, LangChain 1.4.7, langchain-openai 1.7.2, LangGraph 1.2.17. 선언된 최소 Python 3.11에서의 별도 실행 검증은 하지 않았습니다. 정확한 테스트 의존성은 requirements-test.lock을 사용합니다.

## Adversarial review and regression coverage

The review found and fixed three gaps before the initial release:

- Calls injected by mutating `AIMessage.tool_calls` could omit `type`. A JSON
  checkpoint round trip added it, breaking structural idempotence. Cleanup now
  canonicalizes retained calls to `type="tool_call"`.
- Validation compared serialized call IDs but did not compare native content
  block names/arguments with canonical calls. Conflicting representations now
  fail validation; cleanup synchronizes them using the canonical call.
- Direct validation could accept legacy `additional_kwargs.function_call` even
  though cleanup rejected it. Both paths now reject this unsupported protocol.

`test_history_adversarial.py` uses an independent scan-based oracle for retained
call owners, arguments and result precedence. It checks 4,681 short histories,
5,040 permutations of seven events, 100 reproducible longer random histories,
90 ID-budget/late-result scenarios, every cut in a stream-event sequence, and
changing ID budgets across checkpoints. Each accepted history is checked for
input immutability, contiguous matching results, both API serializers, repeated
cleanup and JSON checkpoint round-trip idempotence. These are bounded exhaustive
and seeded tests, not a claim that every possible input was enumerated.

Tests use mock HTTP/SSE rather than the live OpenAI service. Model-specific
capabilities, credentials, context limits and provider availability remain
outside message validation. Serialize runs and checkpoint replacement per thread
in the serving layer; tool retry and duplicate-execution policies belong to the
application.


## Session boundary and module responsibilities

- `history.py`: API-neutral cleanup and structural invariants; no OpenAI import on
  the default `clean_messages(messages)` path.
- `protocols.py`: optional OpenAI serialization validation; explicit legacy
  projection stays available but is not used to migrate session history.
- `session.py`: ApiSessionMiddleware selects history in before_agent using
  persisted session_api and separate request_messages, writes Overwrite, and
  clears the consumed request. It never calls cleanup.
- `middleware.py`: independently cleans selected history and persists changes
  using Overwrite. It validates model completion and converts tool exceptions to ERROR.

The factory returns the original CompiledStateGraph, without an invocation
facade. Request messages must be supplied in request_messages; the messages
channel is accumulated history. Session middleware precedes cleanup middleware.
Resume guards run at model and tool execution because before_agent/before_model
may be skipped when resuming a checkpoint. session_run_active remains true after
interruptions and errors, preventing API changes until completion. This is not a
concurrency lock: serialize runs and external updates per thread. Managed session
fields must not be injected by callers. Standard graph input is checkpointed
before middleware validation, so rejected requests can remain in request_messages;
retry with explicit corrected input. Old snapshots and non-message state remain.
