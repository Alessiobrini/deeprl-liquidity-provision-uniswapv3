import requests

# Test different subgraph endpoints
endpoints = [
    ('Messari', 'https://api.thegraph.com/subgraphs/name/messari/uniswap-v3-ethereum'),
    ('Uniswap Official (deprecated)', 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'),
]

pool_address = '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640'

for name, endpoint in endpoints:
    try:
        query = f'{{pool(id: "{pool_address.lower()}") {{token0 {{symbol}} token1 {{symbol}} feeTier}}}}'
        response = requests.post(endpoint, json={'query': query}, timeout=5)
        print(f'\n{name} ({endpoint}):')
        print(response.json())
    except Exception as e:
        print(f'\n{name}: ERROR - {e}')
