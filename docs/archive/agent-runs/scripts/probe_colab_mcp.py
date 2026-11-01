"""Local protocol probe only: initialize/list tools, never call browser/auth tools."""
import asyncio
import json
import os
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]


async def probe():
    import tomllib
    config = tomllib.loads((ROOT / '.codex/config.toml').read_text())['mcp_servers']['colab']
    params = StdioServerParameters(command=config['command'],
        args=config['args'], cwd=str(ROOT), env=os.environ.copy())
    async with asyncio.timeout(35):
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                initialized = await session.initialize()
                tools = await session.list_tools()
                result = {'initialized': True, 'server': initialized.serverInfo.name,
                          'tools': [tool.name for tool in tools.tools],
                          'browser_opened': False, 'authenticated': False,
                          'scope': 'local stdio handshake only; Codex dynamic tool update not tested'}
                assert 'open_colab_browser_connection' in result['tools']
                (ROOT / 'var/agent_runs/colab-probe.json').write_text(json.dumps(result, indent=2))
                print(json.dumps(result, indent=2))


asyncio.run(probe())
