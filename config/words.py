WORDS = {
'init_log': 'Time: {time}\nCount: {count}',
'user_input': 'User Input: {input}',
'added_data': 'Added {role} Data: {data}',
'history': 'History:\n{history}',
'fired_memories': 'Fired Memories:\n{memories}',
'prompt': 'Prompt:\n{prompt}',
'agent_output': 'Agent Output:\n{response}',
'processing_generation': '===================== Processing generation {generation} =====================',
'search_results': 'Search Results: {results}',
}

def get(key, **kwargs):
    template = WORDS.get(key, '')
    return template.format(**kwargs)

def get_lif_info(child_data, copied_data):
    return\
    f'''{copied_data['id']}: {copied_data['message']['content']}
fire: {child_data['fire']} -> {copied_data['fire']}
v: {child_data['v']} -> {copied_data['v']}
i: {child_data['i']} -> {copied_data['i']}
tau: {child_data['tau']} -> {copied_data['tau']}
spike: {copied_data['spike']}'''