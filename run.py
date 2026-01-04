import os, json, sys
from autogen import ConversableAgent, LLMConfig
from autogen import register_function

from modules.Prompts import get_router_sys_prompt, get_rag_prompt
from modules.Search import get_search_results
from modules.Model import get_llm_config, generate
from modules.Func import termination_msg, get_kg_example_html

# New a output folder
os.makedirs("./output", exist_ok=True)

'''
Model Configuration
'''
# OAI Config
os.environ["OAI_CONFIG_LIST"] = get_llm_config()
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

# Model list
li_models = [
    'mistral-small3.1:24b',
    'qwen3:32b',
    'llama3.3:70b',
    'llama4:16x17b',
    'gpt-oss:20b',
    'gpt-oss:120b',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
    'gpt-3.5-turbo',
    'gpt-4o-mini',
]

# Choose model
model_name = li_models[5]  # Change the index to select different models
print(f"Using model: {model_name}")
llm_config = LLMConfig.from_json(
    path="OAI_CONFIG_LIST",
).where(model=model_name)



'''
Build Agents
'''
# Build an agent to parse user needs
with llm_config:
    # This agent is responsible for parsing user needs and deciding which agent to use for execution
    router_agent = ConversableAgent(
        name="router_agent",
        system_message=get_router_sys_prompt(),
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=termination_msg,
        description="If function calling (tool) returns correctly, say  DONE!; if not completed, ask the executor_agent to use function calling (tool) once more to obtain more information.",
    )

    # An agent responsible for executing function calling (tool)
    executor_agent = ConversableAgent(
        name="executor_agent",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=termination_msg,
        description="An agent that executes function calling (tool) to obtain search results based on user queries."
    )

# Register function calling (tool) to executor_agent
register_function(
    get_search_results,
    caller=router_agent,
    executor=executor_agent,
    name="get_search_results",
    description=f'''Obtain parameters user_intent, hop_type, user_query, query, model_name, search_results, proper_knowledge, num_results = 3, model_name = {model_name}. If Response from calling tool successfully returns, say `DONE!`.''',
)


# Main process
if __name__ == "__main__":
    q = input("Please enter your query: ")
    q_rpl = q.replace("\n", " ")

    while True:
        try:
            # Agents interaction
            chat_result = router_agent.initiate_chat(
                recipient=executor_agent,
                message=q_rpl,
                max_turns=2,
            )

            # Extract function calling result from chat history
            d = None
            with open("./output/debug_chat_result.json", "w", encoding="utf8") as f:
                f.write( json.dumps(chat_result.chat_history, ensure_ascii=False, indent=4) )
            for obj in chat_result.chat_history:
                if obj["role"] == "tool" and obj['name'] == 'executor_agent' and obj["tool_responses"] != None:
                    # Obtain the result of function calling
                    d = eval(obj["tool_responses"][0]["content"])

                    # Termination condition
                    break
            
            # Check if function calling result is obtained
            if d is None:
                print("No function calling result found.")
                continue

            # Obtain customized user prompt
            d['user_query'] = q
            with open("./output/debug_search_result.json", "w", encoding="utf8") as f:
                f.write( json.dumps(d, ensure_ascii=False, indent=4) )
            user_prompt = get_rag_prompt(d)

            # For Agentic IR, perform generative response
            generated_text = generate(user_prompt, model_name)

            # Output results
            with open(f"./output/search_result.json", "w", encoding="utf8") as f:
                f.write(json.dumps(d, ensure_ascii=False, indent=None))
            with open(f"./output/user_prompt.txt", "w", encoding="utf8") as f:
                f.write(user_prompt + generated_text)
            with open(f"./output/kg_example.html", "w", encoding="utf8") as f:
                li_triplets = []
                for pk in d['proper_knowledge']:
                    if 'triplets' in pk:
                        li_triplets.extend( pk['triplets'] )
                f.write(get_kg_example_html( str(li_triplets) ))

            break

        except Exception as e:
            print(f"Error: {e}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)