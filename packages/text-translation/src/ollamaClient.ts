import type { LlmChatCompletionRequestBody, ChatCompletionResponse } from './contracts';

export type OllamaModelInfo = {
  name: string;
  size?: number;
  modifiedAt?: string;
};

export type OllamaConnectionInfo = {
  connected: true;
  version?: string;
};

export type OllamaClientErrorCode =
  | 'OLLAMA_INVALID_URL'
  | 'OLLAMA_CONNECTION_FAILED'
  | 'OLLAMA_HTTP_ERROR'
  | 'OLLAMA_INVALID_RESPONSE';

export class OllamaClientError extends Error {
  readonly code: OllamaClientErrorCode;
  readonly status?: number;
  readonly retryable?: boolean;

  constructor(
    message: string,
    options: {
      code: OllamaClientErrorCode;
      status?: number;
      retryable?: boolean;
      cause?: unknown;
    },
  ) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = 'OllamaClientError';
    this.code = options.code;
    this.status = options.status;
    this.retryable = options.retryable;
  }
}

/**
 * Ollama is intentionally restricted to loopback hosts. Requests originate in
 * the extension background, so accepting arbitrary hosts here would turn the
 * extension message bridge into a generic private-network fetch proxy.
 */
export function normalizeOllamaServerUrl(value: string): string {
  let url: URL;
  try {
    url = new URL(value.trim());
  } catch (cause) {
    throw new OllamaClientError('Ollama 服务地址格式无效', {
      code: 'OLLAMA_INVALID_URL',
      cause,
    });
  }

  const hostname = url.hostname.replace(/^\[|\]$/gu, '').toLowerCase();
  const loopback = hostname === 'localhost'
    || hostname === '127.0.0.1'
    || hostname === '::1';
  const path = url.pathname.replace(/\/+$/u, '');

  if (
    url.protocol !== 'http:'
    || !loopback
    || url.username
    || url.password
    || url.search
    || url.hash
    || (path !== '' && path !== '/v1')
  ) {
    throw new OllamaClientError(
      'Ollama 仅允许使用本机 HTTP 地址，例如 http://127.0.0.1:11434',
      { code: 'OLLAMA_INVALID_URL' },
    );
  }

  return url.origin;
}

function readErrorDetail(text: string): string {
  if (!text) return '';
  try {
    const value = JSON.parse(text) as { error?: unknown; message?: unknown };
    if (typeof value.error === 'string') return value.error;
    if (value.error && typeof value.error === 'object' && 'message' in value.error
      && typeof value.error.message === 'string') return value.error.message;
    if (typeof value.message === 'string') return value.message;
  } catch {
    // Preserve plain-text Ollama errors below.
  }
  return text.slice(0, 500);
}

export type OllamaFetch = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>;

export type OllamaClient = {
  listModels(baseUrl: string, signal?: AbortSignal): Promise<OllamaModelInfo[]>;
  testConnection(baseUrl: string, signal?: AbortSignal): Promise<OllamaConnectionInfo>;
  chatCompletions(
    baseUrl: string,
    body: LlmChatCompletionRequestBody,
    signal?: AbortSignal,
  ): Promise<ChatCompletionResponse>;
};

export function createOllamaClient(fetcher: OllamaFetch = (...args) => fetch(...args)): OllamaClient {
  async function request(url: string, init: RequestInit): Promise<Response> {
    try {
      return await fetcher(url, init);
    } catch (cause) {
      if (cause instanceof Error && cause.name === 'AbortError') throw cause;
      throw new OllamaClientError(
        '无法连接本机 Ollama。请确认 Ollama 正在运行，并检查 Chrome 是否允许扩展访问本地网络。',
        {
          code: 'OLLAMA_CONNECTION_FAILED',
          retryable: true,
          cause,
        },
      );
    }
  }

  async function readJson(response: Response, operation: string): Promise<unknown> {
    const responseText = await response.text();
    if (!response.ok) {
      const detail = readErrorDetail(responseText);
      throw new OllamaClientError(
        detail ? `Ollama ${operation}失败：${detail}` : `Ollama ${operation}失败（HTTP ${response.status}）`,
        {
          code: 'OLLAMA_HTTP_ERROR',
          status: response.status,
          retryable: response.status === 429 || response.status >= 500,
        },
      );
    }
    try {
      return JSON.parse(responseText) as unknown;
    } catch (cause) {
      throw new OllamaClientError(`Ollama ${operation}返回了无效 JSON`, {
        code: 'OLLAMA_INVALID_RESPONSE',
        status: response.status,
        cause,
      });
    }
  }

  return {
    async listModels(baseUrl, signal) {
      const origin = normalizeOllamaServerUrl(baseUrl);
      const response = await request(`${origin}/api/tags`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
        ...(signal ? { signal } : {}),
      });
      const data = await readJson(response, '模型列表请求');
      if (!data || typeof data !== 'object' || !Array.isArray((data as { models?: unknown }).models)) {
        throw new OllamaClientError('Ollama 模型列表响应缺少 models 数组', {
          code: 'OLLAMA_INVALID_RESPONSE',
          status: response.status,
        });
      }
      return (data as { models: unknown[] }).models.flatMap((item): OllamaModelInfo[] => {
        if (!item || typeof item !== 'object') return [];
        const model = item as Record<string, unknown>;
        const name = typeof model.name === 'string'
          ? model.name
          : typeof model.model === 'string' ? model.model : '';
        if (!name) return [];
        return [{
          name,
          ...(typeof model.size === 'number' ? { size: model.size } : {}),
          ...(typeof model.modified_at === 'string' ? { modifiedAt: model.modified_at } : {}),
        }];
      });
    },

    async testConnection(baseUrl, signal) {
      const origin = normalizeOllamaServerUrl(baseUrl);
      const response = await request(`${origin}/api/version`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
        ...(signal ? { signal } : {}),
      });
      const data = await readJson(response, '连接测试');
      if (!data || typeof data !== 'object') {
        throw new OllamaClientError('Ollama 连接测试返回了无效数据', {
          code: 'OLLAMA_INVALID_RESPONSE',
          status: response.status,
        });
      }
      const version = typeof (data as { version?: unknown }).version === 'string'
        ? (data as { version: string }).version
        : undefined;
      return { connected: true, ...(version ? { version } : {}) };
    },

    async chatCompletions(baseUrl, body, signal) {
      const origin = normalizeOllamaServerUrl(baseUrl);
      const response = await request(`${origin}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(body),
        ...(signal ? { signal } : {}),
      });
      const data = await readJson(response, '聊天补全请求');
      if (!data || typeof data !== 'object') {
        throw new OllamaClientError('Ollama 聊天补全响应不是有效对象', {
          code: 'OLLAMA_INVALID_RESPONSE',
          status: response.status,
        });
      }
      return data as ChatCompletionResponse;
    },
  };
}

const defaultOllamaClient = createOllamaClient();

export function listOllamaModels(baseUrl: string, signal?: AbortSignal): Promise<OllamaModelInfo[]> {
  return defaultOllamaClient.listModels(baseUrl, signal);
}

export function testOllamaConnection(baseUrl: string, signal?: AbortSignal): Promise<OllamaConnectionInfo> {
  return defaultOllamaClient.testConnection(baseUrl, signal);
}

export function requestOllamaChatCompletion(
  baseUrl: string,
  body: LlmChatCompletionRequestBody,
  signal?: AbortSignal,
): Promise<ChatCompletionResponse> {
  return defaultOllamaClient.chatCompletions(baseUrl, body, signal);
}

export function resolveOllamaChatCompletionsEndpoint(baseUrl: string): string {
  return `${normalizeOllamaServerUrl(baseUrl)}/v1/chat/completions`;
}
