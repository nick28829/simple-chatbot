# Intents
- get a recall
- get general work experience
- get details about working experience
    - details about job
    - details about company
- get details about academic career
- get details about skills

# Example Context
```json
{
    "follow_up": true | false,
    "intent_args": [
        {
            "name": "arg",
            "value": "val" 
        }
    ],
    "intent": "someIntent",
    "missing_args": [
        "name"
    ],
    "intent_history": [
        "intent1", 
        "intent2"
    ]
}
```

# TODO
[] replace args and placeholder with name
[] in weights, include IDF with all sample sentences as docs
[] add more actions
[] serve frontend with python server
[] make frontend a bit prettier