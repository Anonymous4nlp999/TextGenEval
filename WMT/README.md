data format

To save space, the scores are stored in string
```json
{
    doc_id: {
        'src': ...,
        'ref': ...,
        'better': {
            'sys': ...,
            'scores': ...
        },
        'worse': {
            'sys': ...,
            'scores': ...
        }
    }
}
```

To run BLEURT:
```
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
```

Cannot run BLEURT and COMET..need to see why