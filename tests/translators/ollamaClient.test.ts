import { afterEach, describe, expect, it, vi } from 'vitest';
import { createOllamaClient, normalizeOllamaServerUrl } from '../../packages/text-translation/src/ollamaClient';

afterEach(() => vi.restoreAllMocks());

describe('normalizeOllamaServerUrl', () => {
  it.each([
    ['http://127.0.0.1:11434', 'http://127.0.0.1:11434'],
    ['http://localhost:11434/v1/', 'http://localhost:11434'],
    ['http://[::1]:11434/v1', 'http://[::1]:11434'],
  ])('normalizes %s', (value, expected) => {
    expect(normalizeOllamaServerUrl(value)).toBe(expected);
  });

  it.each([
    'https://127.0.0.1:11434',
    'http://192.168.1.20:11434',
    'http://example.com:11434',
    'http://user:pass@127.0.0.1:11434',
    'http://127.0.0.1:11434/api',
    'http://127.0.0.1:11434/?token=x',
  ])('rejects unsafe or unsupported URL %s', (value) => {
    expect(() => normalizeOllamaServerUrl(value)).toThrow();
  });
});

describe('Ollama client', () => {
  it('discovers local models from /api/tags', async () => {
    const fetcher = vi.fn(async () => Response.json({
      models: [
        { name: 'qwen3.5:9b', size: 6600000000, modified_at: '2026-11-01T00:00:00Z' },
        { model: 'llama3.2:3b' },
        { name: '' },
      ],
    }));
    const client = createOllamaClient(fetcher);
    await expect(client.listModels('http://127.0.0.1:11434/v1')).resolves.toEqual([
      { name: 'qwen3.5:9b', size: 6600000000, modifiedAt: '2026-11-01T00:00:00Z' },
      { name: 'llama3.2:3b' },
    ]);
    expect(fetcher).toHaveBeenCalledWith('http://127.0.0.1:11434/api/tags', expect.objectContaining({ method: 'GET' }));
  });

  it('tests connection using /api/version', async () => {
    const fetcher = vi.fn(async () => Response.json({ version: '0.12.0' }));
    const client = createOllamaClient(fetcher);
    await expect(client.testConnection('http://localhost:11434')).resolves.toEqual({ connected: true, version: '0.12.0' });
    expect(fetcher).toHaveBeenCalledWith('http://localhost:11434/api/version', expect.any(Object));
  });

  it('uses OpenAI-compatible chat completions without an API key', async () => {
    const response = { choices: [{ message: { content: '本地译文' } }] };
    const fetcher = vi.fn(async () => Response.json(response));
    const client = createOllamaClient(fetcher);
    const body = { model: 'qwen3.5:9b', messages: [{ role: 'user' as const, content: 'translate' }] };
    await expect(client.chatCompletions('http://127.0.0.1:11434/v1', body)).resolves.toEqual(response);
    const [url, init] = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('http://127.0.0.1:11434/v1/chat/completions');
    expect(init.method).toBe('POST');
    expect(init.headers).toEqual({ 'Content-Type': 'application/json', Accept: 'application/json' });
    expect(init.headers).not.toHaveProperty('Authorization');
    expect(JSON.parse(init.body as string)).toEqual(body);
  });

  it('surfaces missing-model HTTP errors with status', async () => {
    const fetcher = vi.fn(async () => new Response(JSON.stringify({ error: 'model not found' }), { status: 404 }));
    const client = createOllamaClient(fetcher);
    await expect(client.chatCompletions('http://127.0.0.1:11434', {
      model: 'missing-model', messages: [],
    })).rejects.toMatchObject({ name: 'OllamaClientError', code: 'OLLAMA_HTTP_ERROR', status: 404 });
  });
});
