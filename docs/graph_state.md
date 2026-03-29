# LangGraph State Graph

This diagram reflects the exact execution order and state mutations in `backend/src/research_agent/graph/builder.py`.

## Diagram (Mermaid)
```mermaid
flowchart TD
  START([START]) --> PREP[prepare_mode]
  PREP --> RETR[retrieve]
  RETR --> RERANK[rerank]
  RERANK --> DRAFT[draft_answer]
  DRAFT --> VALID[validate_answer]
  VALID --> FINAL[finalize_answer]
  FINAL --> END([END])

  subgraph "Process Order"
    P1["1. START: Graph invoked with session_id, mode, message, paper_ids, review_paper_id, history"]
    P2["2. prepare_mode: set mode_instructions, debug.prepared_mode, debug.paper_count"]
    P3["3. retrieve: mode-aware retrieval, collect candidates, retrieval debug"]
    P4["4. rerank: lexical + section-signal rerank, comparator balancing, citations"]
    P5["5. draft_answer: generate first grounded draft with citations"]
    P6["6. validate_answer: second-pass factual validator revises unsupported claims"]
    P7["7. finalize_answer: select final answer and attach citations/debug"]
  end

  START -.-> P1
  PREP -.-> P2
  RETR -.-> P3
  RERANK -.-> P4
  DRAFT -.-> P5
  VALID -.-> P6
  FINAL -.-> P7
```

## Exact Process Order
1. START: `runtime.chat()` invokes the graph with the initial state.
2. prepare_mode: `_prepare_mode_step` sets `mode_instructions` and debug fields.
3. retrieve: `_retrieve_step` selects retrieval strategy by mode and returns candidate `retrieved_documents` with retrieval debug.
4. rerank: `_rerank_step` reorders candidates using lexical overlap, section-signal boosts, and paper balancing for comparator mode.
5. draft_answer: `_draft_answer_step` validates mode constraints and generates an initial grounded draft.
6. validate_answer: `_validate_answer_step` runs a factual verifier pass that revises unsupported claims.
7. finalize_answer: `_finalize_answer_step` selects the best final answer, returns `answer`, `citations`, and debug.
8. END: graph returns final state to the API.

## State Fields (GraphState)
- session_id
- mode
- message
- paper_ids
- review_paper_id
- history
- mode_instructions
- retrieved_documents
- draft_answer
- validated_answer
- validation_issues
- answer
- citations
- debug
