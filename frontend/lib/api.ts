const base = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export async function getJSON<T>(path: string) {
  const res = await fetch(`${base}${path}`);
  if (!res.ok) throw new Error(`GET ${path} ${res.status}`);
  return res.json() as Promise<T>;
}

export async function postJSON<T>(path: string, body: any) {
  const res = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} ${res.status}`);
  return res.json() as Promise<T>;
}

export function pdfUrl(analysisId: string) { return `${base}/api/report/pdf/${analysisId}`; }
export const apiBase = base;
