import os, sys

# Get the system prompt for the router agent
def get_router_sys_prompt():
    return '''\
The user's question is named user_query.
Do not modify user_query.
Do not answer user_query directly. Only decide hop_type and produce query/sub-queries for tool use.

===================================
DEFINITIONS (STRICT)

Atomic fact:
- A single factual statement that can be retrieved directly without needing to resolve an intermediate unknown entity first.
Example atomic facts:
- "Apple Inc. was founded in 1976."
- "Albert Einstein's nationality was German."
- "Tokyo is the capital of Japan."
- "The first iPhone was released in 2007."
- "The Mona Lisa was painted by Leonardo da Vinci."

Hop count:
- The minimal number of atomic facts required to answer user_query.

SINGLE-HOP:
- hop_count == 1
- The target answer is a direct attribute of a clearly specified entity (or direct lookup).
- No intermediate entity needs to be identified first.

MULTI-HOP:
- hop_count >= 2, especially when an intermediate entity/answer must be resolved first.

HARD RULE (override):
If user_query contains an INDIRECT / NESTED reference such as:
- "the X who/that ...", "the X of the Y who/that ...",
- "the country where [person] was born",
- "the year when [event related to another event/person] ...",
- any pattern where the entity asked about is defined by another relation,
THEN classify as MULTI-HOP.

Conservative rule:
- If uncertain, classify as MULTI-HOP.

===================================
DECISION PROCEDURE (must follow)

1) Identify the final target requested (e.g., year, person, place, number).
2) Ask: Is the target attribute attached to an entity that is fully specified in the question?
   - If YES and only one attribute lookup is needed -> SINGLE-HOP.
   - If NO, you must first identify an intermediate entity/answer -> MULTI-HOP.
3) For MULTI-HOP, produce minimal sub-questions that are each single-hop.

===================================
EXAMPLES (include nested-reference cues)

Single-hop:
- Q: What is the capital of France?
  hop_count=1 (France -> capital)

Multi-hop:
- Q: In which year was the inventor of the telephone born?
  hop_count=2
  SubQ1: Who invented the telephone?
  SubQ2: In which year was [ANSWER of SubQ1] born?

- Q: What is the capital of the country where the director of Argo was born?
  hop_count=3
  SubQ1: Who directed Argo?
  SubQ2: Where was [director] born? (country)
  SubQ3: What is the capital of [country]?

===================================
TOOL USE

Use the tool via function calling: get_search_results()

If MULTI-HOP:
- Provide: user_intent, hop_type="multi-hop", user_query, num_results, model_name,
  and an array of sub_questions (each should be simple, single-hop, standalone).

If SINGLE-HOP:
- Provide: user_intent, hop_type="single-hop", user_query, query (single query), num_results, model_name.

Output MUST be only the tool call payload.
'''


# Get the summarization prompt
def get_summarization_prompt(q, context) -> str:
    # Get the summarization
    prompt = f'''\
QUESTION:
{q}

========================================

CONTEXT:
{context}

========================================

RULES:
1. Pick 5 paragraphs ONLY from CONTEXT.
2. Paragraphs MUST be able to answer the QUESTION, or provide answers which can be used to answer the QUESTION.
3. Summarize ONLY the selected paragraphs.
4. Elements must be separated by commas.
5. For any double quotes in the SUMMARY content:
- Replace \ with \\
- Replace " with \"
- Replace any newline with \n
- Replace any tab with \t

========================================

RESPONSED FORMAT:
Return ONLY a valid JSON array of three strings.  
Example: ["SUMMARY 1", "SUMMARY 2", "SUMMARY 3", "SUMMARY 4", "SUMMARY 5"]

At the end, verify that your output is valid JSON and matches the example format exactly.

========================================

OUTPUT:'''
    return prompt


# Get the prompt for extracting knowledge triples
def get_triplets_prompt(d_proper_knowledge):
    li_text = []
    for references in d_proper_knowledge['references']:
        li_text.append(references['text'])

    references = "\n".join(li_text)

    prompt = f'''\
Triples describe relationships between entities and consist of three elements. 
In computer science, triples are commonly used to represent data in relational databases. 
A typical triple contains three components: Subject, Predicate, and Object. They form the basic elements of a Knowledge Graph.

=====================================================================================

Format:
triples = [
    ["Subject", "Predicate", "Object"],
    ["Subject", "Predicate", "Object"],
    ["Subject", "Predicate", "Object"],
    ["Subject", "Predicate", "Object"],
    ["Subject", "Predicate", "Object"],
    ["Subject", "Predicate", "Object"],
    ["Subject", "Predicate", "Object"],
    ["Subject", "Predicate", "Object"],
    ["Subject", "Predicate", "Object"],
    ["Subject", "Predicate", "Object"],
    ["Subject", "Predicate", "Object"],
    ["Subject", "Predicate", "Object"],
    ...
]
...and so on.

Instructions:
1. If multiple Subjects refer to the same entity, unify them into a single Subject to ensure correct relationships and reduce unnecessary duplication.
2. If a single sentence or text paragraph contains a Subject associated with more than one Predicate and Object, list all such relationships.
3. Do not answer any questions or provide explanations — only extract and list the triples.
4. Elements must be separated by commas.
5. For any double quotes in the Subject/Predicate/Object content:
- Replace \ with \\
- Replace " with \"
- Replace any newline with \n
- Replace any tab with \t

=====================================================================================

Extract all triples from the following text:
{references}

At the end, verify that your output is valid LIST OF TRIPLES and matches the example format exactly.

Output:'''
    return prompt


# Get the prompt for LLM to rerank results
def get_rerank_results_by_llm(ranks, q) -> str:
    prompt = f'''\
Re-rank the following search results based on their relevance to the user's query.

Query:
{q}

=========================

Search results (list of dictionaries):
{ranks}

=========================

Instruction:
The list is reordered based on relevance to the user's query by yourself, but the originally provided scores are retained.
The list will be used to calculate cumulative gain (CG), discounted cumulative gain (DCG), and normalized discounted cumulative gain (NDCG).
Keep the original relevance scores, URLs, and summaries.
Do not include any explanations, comments, or extra text.
Do not include any characters/strings that would break JSON format.
Only output the final result as a valid JSON array.
You MUST follow the example format below exactly.
Avoid this issue: "json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes".
You MUST ensure the output is valid JSON.

=========================

FORMAT:
[
    {{
        "corpus_id": original_id,
        "score": original_score,
        "text": "original_summary",
        "url": "original_url"
    }},
    ...
]

Ensure the output is valid JSON.

Output: '''
    return prompt




# Integrate retrieval results into the user prompt
def get_rag_prompt(d: dict):
    try:
        knowledge = ""
        for d_ in d['proper_knowledge']:
            # Organize relationships between triplets
            str_triplets = ""
            if 'triplets' in d_:
                triplets = d_['triplets']
                for t in triplets:
                    # Convert each triple to string format
                    if len(t) < 3:
                        continue
                    str_triplets += f'`Subject: "{t[0]}", Predicate: "{t[1]}", Object: "{t[2]}"`\n'

            # Organize references
            for reference in d_['references']:
                # Organize knowledge string
                knowledge += f'''\
----------------------------------
Sub-question: {d_['q']}
Reference: {reference['text']}
Relevance Score: {reference['score']}
Source: {reference['url']}
'''

            knowledge += f'''\
----------------------------------
Triples (Knowledge Graph): 
{str_triplets}'''

        # Organize user prompt
        context = f'''\
Original Question:
{d['user_query']}

=========================

Please refer to the following references and triples to answer the question:
{knowledge}

=========================

Notes:
1. The answer must begin with the correct option from the question choices.
2. If the question is multi-choice, choose the best answer (A), (B), (C) or (D) based on your understanding of the question.
3. If the reference knowledges are unavailable or unreasonable, you need to answer with your own understanding.
4. Answer the question in English.
5. After answering the question, output a JSON object named contribution with two integer fields: references_pct and triples_pct.
- Constraints: each is 0–100, and references_pct + triples_pct = 100.
- Do not include additional keys or commentary.
- e.g. "contribution": {{"references_pct": percentage, "triples_pct": percentage}}

=========================

Answer:'''
        return context
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # Name of the error event
        print(exc_type, fname, exc_tb.tb_lineno)

        # Code where the error occurred
        print(f"Error in Prompts: {e}")