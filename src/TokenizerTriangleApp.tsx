import React, { useMemo, useRef, useState } from "react";

type Weights = { acoustic: number; semantic: number; linguistic: number };

type Tokenizer = {
  id: string;
  name: string;
  url: string;
  group: "acoustic" | "semantic" | "linguistic" | "hybrid";
  weights: Weights;
  frameRate: string;
  encoder: string;
  decoder: string;
  rep: string;
  quant: string;
  objectives: string;
  aux: string;
  notes?: string;
};

function normalizeWeights(w: Weights): Weights {
  const s = w.acoustic + w.semantic + w.linguistic;
  if (s <= 0) return { acoustic: 1, semantic: 0, linguistic: 0 };
  return { acoustic: w.acoustic / s, semantic: w.semantic / s, linguistic: w.linguistic / s };
}

function baryToXY(
  w: Weights,
  A: { x: number; y: number },
  S: { x: number; y: number },
  L: { x: number; y: number }
) {
  const wn = normalizeWeights(w);
  const x = wn.acoustic * A.x + wn.semantic * S.x + wn.linguistic * L.x;
  const y = wn.acoustic * A.y + wn.semantic * S.y + wn.linguistic * L.y;
  return { x, y };
}

const DATA: Tokenizer[] = [
  {
    id: "encodec",
    name: "EnCodec (2023)",
    url: "https://arxiv.org/abs/2210.13438",
    group: "acoustic",
    weights: { acoustic: 0.90, semantic: 0.04, linguistic: 0.06 },
    frameRate: "75, 150 Hz",
    encoder: "CNN+RNN",
    decoder: "CNN",
    rep: "T",
    quant: "RVQ",
    objectives: "GAN, Feat, Rec, VQ",
    aux: "–",
    notes: "High‑fidelity neural audio codec; multi‑domain (speech/music/audio).",
  },
  {
    id: "dac",
    name: "DAC (2023)",
    url: "https://arxiv.org/abs/2306.06546",
    group: "acoustic",
    weights: { acoustic: 0.96, semantic: 0.02, linguistic: 0.02 },
    frameRate: "75 Hz",
    encoder: "CNN",
    decoder: "CNN",
    rep: "T",
    quant: "RVQ",
    objectives: "GAN, Feat, Rec, VQ",
    aux: "–",
    notes: "Waveform‑domain codec focusing on perceptual quality via GAN + feature matching.",
  },
  {
    id: "wavtokenizer",
    name: "WavTokenizer (2024)",
    url: "https://arxiv.org/abs/2408.16532",
    group: "acoustic",
    weights: { acoustic: 0.90, semantic: 0.08, linguistic: 0.02 },
    frameRate: "40, 75 Hz",
    encoder: "CNN+Transformer",
    decoder: "CNN+Transformer",
    rep: "T",
    quant: "SVQ",
    objectives: "GAN, Feat, Rec, VQ",
    aux: "–",
    notes: "Acoustic tokens with single‑codebook quantization; LLM‑friendly token rates.",
  },
  {
    id: "speechtokenizer",
    name: "SpeechTokenizer (2024)",
    url: "https://arxiv.org/abs/2308.16692",
    group: "semantic",
    weights: { acoustic: 0.45, semantic: 0.50, linguistic: 0.05 },
    frameRate: "50 Hz",
    encoder: "CNN+RNN",
    decoder: "CNN",
    rep: "T",
    quant: "RVQ",
    objectives: "GAN, Rec, Feat, VQ",
    aux: "SD",
    notes: "Semantic distillation guides early codebooks; tokens carry phonetic content.",
  },
  {
    id: "mimi",
    name: "Mimi (2024)",
    url: "https://arxiv.org/abs/2410.00037",
    group: "semantic",
    weights: { acoustic: 0.35, semantic: 0.58, linguistic: 0.07 },
    frameRate: "12.5 Hz",
    encoder: "CNN+Transformer",
    decoder: "CNN+Transformer",
    rep: "T",
    quant: "RVQ",
    objectives: "GAN, Feat, Rec, VQ",
    aux: "SD",
    notes: "Very low token rate; SSL‑guided semantics with high‑fidelity reconstruction.",
  },
  {
    id: "xcodec",
    name: "X‑Codec (2025)",
    url: "https://arxiv.org/abs/2408.17175",
    group: "semantic",
    weights: { acoustic: 0.35, semantic: 0.65, linguistic: 0.0 },
    frameRate: "50 Hz",
    encoder: "CNN (dual‑path)",
    decoder: "CNN",
    rep: "T",
    quant: "RVQ",
    objectives: "GAN, Rec, VQ",
    aux: "SD",
    notes: "Semantic guidance via SSL features; dual semantic/acoustic paths before RVQ.",
  },
  {
    id: "xcodec2",
    name: "X‑codec2 (2025)",
    url: "https://arxiv.org/pdf/2502.04128",
    group: "semantic",
    weights: { acoustic: 0.33, semantic: 0.55, linguistic: 0.12 },
    frameRate: "50 Hz",
    encoder: "Transformer + Semantic encoder",
    decoder: "Transformer (Vocos‑style)",
    rep: "T‑F",
    quant: "FSQ",
    objectives: "GAN, Rec",
    aux: "SD",
    notes: "Unifies semantic + acoustic streams; FSQ for LLM‑friendly discrete tokens.",
  },
  {
    id: "maskgct",
    name: "MaskGCT (2024)",
    url: "https://arxiv.org/pdf/2409.00750",
    group: "semantic",
    weights: { acoustic: 0.20, semantic: 0.80, linguistic: 0.00 },
    frameRate: "49 Hz",
    encoder: "CNN",
    decoder: "CNN",
    rep: "T‑F",
    quant: "SVQ",
    objectives: "Rec, VQ",
    aux: "SD",
    notes: "Semantic codec distilled from W2v‑BERT; single‑codebook VQ‑VAE.",
  },
  {
    id: "lscodec",
    name: "LSCodec (2025)",
    url: "https://arxiv.org/abs/2410.15764",
    group: "hybrid",
    weights: { acoustic: 0.50, semantic: 0.50, linguistic: 0.00 },
    frameRate: "25, 50 Hz",
    encoder: "CNN",
    decoder: "CNN",
    rep: "T",
    quant: "SVQ",
    objectives: "GAN, Feat, Rec",
    aux: "Dis, SD",
    notes: "Ultra‑low bitrate; disentangles timbre from tokens via multi‑stage training.",
  },
  {
    id: "s3",
    name: "S3 (CosyVoice, 2024)",
    url: "https://fun-audio-llm.github.io/pdf/CosyVoice_v1.pdf",
    group: "linguistic",
    weights: { acoustic: 0.05, semantic: 0.05, linguistic: 0.90 },
    frameRate: "–",
    encoder: "Transformer",
    decoder: "Transformer",
    rep: "T",
    quant: "FSQ",
    objectives: "Diff",
    aux: "SST",
    notes: "Semantic‑to‑linguistic supervision for discrete tokens; diffusion objectives.",
  },
  {
    id: "tadicodec",
    name: "TaDiCodec (2025)",
    url: "https://arxiv.org/pdf/2508.16790",
    group: "linguistic",
    weights: { acoustic: 0.10, semantic: 0.10, linguistic: 0.80 },
    frameRate: "6.25 Hz",
    encoder: "Transformer",
    decoder: "Transformer (Diffusion)",
    rep: "T‑F",
    quant: "FSQ (BSQ)",
    objectives: "Diff",
    aux: "SST* (text‑aware diffusion)",
    notes: "Text‑aware diffusion codec; extreme compression for SLM/TTS tokens.",
  },
  {
    id: "minimax",
    name: "MiniMax‑Speech (2025)",
    url: "https://arxiv.org/pdf/2505.07916",
    group: "linguistic",
    weights: { acoustic: 0.25, semantic: 0.05, linguistic: 0.70 },
    frameRate: "25 Hz",
    encoder: "AR Transformer (tokenizer as VQ‑VAE)",
    decoder: "Transformer",
    rep: "T",
    quant: "VQ‑VAE",
    objectives: "Rec",
    aux: "SST (CTC)",
    notes: "CTC supervision used to preserve semantics under high compression.",
  },
  {
    id: "facodec",
    name: "FACodec (2024)",
    url: "https://arxiv.org/abs/2403.03100",
    group: "linguistic",
    weights: { acoustic: 0.20, semantic: 0.40, linguistic: 0.40 },
    frameRate: "80 Hz",
    encoder: "CNN+RNN",
    decoder: "CNN+RNN",
    rep: "T",
    quant: "GRVQ / FVQ",
    objectives: "GAN, Feat, Rec, VQ",
    aux: "Dis + supervised (phonemes/F0/speaker)",
    notes: "Factorized tokens: Content, Prosody, Timbre, Acoustic details; GRL losses.",
  },
];

const GROUP_STYLE: Record<Tokenizer["group"], { fill: string; stroke: string; chip: string }> = {
  acoustic: { fill: "#60a5fa", stroke: "#1d4ed8", chip: "bg-blue-100 text-blue-800" },
  semantic: { fill: "#34d399", stroke: "#065f46", chip: "bg-emerald-100 text-emerald-800" },
  linguistic: { fill: "#f472b6", stroke: "#9d174d", chip: "bg-pink-100 text-pink-800" },
  hybrid: { fill: "#fbbf24", stroke: "#b45309", chip: "bg-amber-100 text-amber-800" },
};

export default function TokenizerTriangleApp() {
  const [query, setQuery] = useState("");
  const [activeGroups, setActiveGroups] = useState<Record<Tokenizer["group"], boolean>>({
    acoustic: true,
    semantic: true,
    linguistic: true,
    hybrid: true,
  });
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [lockedId, setLockedId] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const isEmbedded = typeof window !== "undefined" && new URLSearchParams(window.location.search).get("embedded") === "true";

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return DATA.filter((d) => activeGroups[d.group]).filter((d) =>
      !q
        ? true
        : [
            d.name,
            d.encoder,
            d.decoder,
            d.quant,
            d.objectives,
            d.aux,
            d.notes ?? "",
          ]
            .join(" ")
            .toLowerCase()
            .includes(q)
    );
  }, [query, activeGroups]);

  const width = 900;
  const height = 720;

  const A = { x: 80, y: height - 80 };
  const S = { x: width - 80, y: height - 80 };
  const L = { x: width / 2, y: 80 };

  const points = filtered.map((d) => ({
    d,
    ...baryToXY(d.weights, A, S, L),
  }));

  const current = points.find((p) => p.d.id === (lockedId ?? hoverId));

  const toggleGroup = (g: Tokenizer["group"]) =>
    setActiveGroups((st) => ({ ...st, [g]: !st[g] }));

  return (
    <div className={isEmbedded ? "w-screen h-screen bg-slate-950 text-slate-100 p-0 m-0" : "w-full min-h-screen bg-slate-950 text-slate-100 p-6"}>
      <div className={isEmbedded ? "w-screen h-screen" : "max-w-6xl mx-auto space-y-4"}>
        <header className={isEmbedded ? "hidden" : "flex flex-col gap-2"}>
          <h1 className="text-2xl md:text-3xl font-semibold">Speech Tokenizers: Acoustic • Semantic • Linguistic</h1>
          <p className="text-slate-300 text-sm md:text-base">
            Hover a point to preview details. Click a point to lock the tooltip. Use the chips to filter by group or search by name/architecture.
          </p>
        </header>

        <div className={isEmbedded ? "hidden" : "flex flex-wrap items-center gap-3"}>
          {(Object.keys(GROUP_STYLE) as Array<Tokenizer["group"]>).map((g) => (
            <button
              key={g}
              onClick={() => toggleGroup(g)}
              className={`px-3 py-1.5 rounded-full text-sm font-medium border ${
                activeGroups[g]
                  ? `${GROUP_STYLE[g].chip} border-transparent`
                  : `bg-transparent border-slate-600 text-slate-300`
              }`}
            >
              {g[0].toUpperCase() + g.slice(1)}
            </button>
          ))}

          <div className="ml-auto flex items-center gap-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search tokenizers…"
              className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 w-64 focus:outline-none focus:ring-2 focus:ring-sky-500"
            />
          </div>
        </div>

        <div className={isEmbedded ? "w-full h-full" : "bg-slate-900/60 rounded-2xl p-4 shadow-lg border border-slate-800"}>
          <div className="relative">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${width} ${height}`}
              className={isEmbedded ? "w-full h-full" : "w-full h-[72vh] max-h-[720px]"}
            >
              <defs>
                <linearGradient id="tri" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#0f172a" />
                  <stop offset="100%" stopColor="#0b1220" />
                </linearGradient>
              </defs>
              <polygon
                points={`${A.x},${A.y} ${S.x},${S.y} ${L.x},${L.y}`}
                fill="url(#tri)"
                stroke="#334155"
                strokeWidth={2}
              />

              {[0.25, 0.5, 0.75].map((t, i) => {
                const interp = (P: any, Q: any, r: number) => ({ x: P.x + (Q.x - P.x) * r, y: P.y + (Q.y - P.y) * r });
                const AL = interp(A, L, t);
                const SL = interp(S, L, t);
                return (
                  <g key={i}>
                    <line x1={AL.x} y1={AL.y} x2={SL.x} y2={SL.y} stroke="#1f2937" strokeDasharray="4 6" />
                  </g>
                );
              })}

              <g fontFamily="ui-sans-serif, system-ui" fontWeight={600}>
                <text x={A.x - 10} y={A.y + 28} textAnchor="start" className="fill-blue-300">Acoustic</text>
                <text x={S.x + 10} y={S.y + 28} textAnchor="end" className="fill-emerald-300">Semantic</text>
                <text x={L.x} y={L.y - 16} textAnchor="middle" className="fill-pink-300">Linguistic</text>
              </g>

              {points.map((p) => {
                const style = GROUP_STYLE[p.d.group];
                return (
                  <g key={p.d.id}
                     onMouseEnter={() => setHoverId(p.d.id)}
                     onMouseLeave={() => setHoverId((id) => (lockedId ? id : null))}
                     onClick={() => setLockedId((id) => (id === p.d.id ? null : p.d.id))}
                     className="cursor-pointer">
                    <circle cx={p.x} cy={p.y} r={8} fill={style.fill} stroke={style.stroke} strokeWidth={2} />
                    <text x={p.x + 12} y={p.y - 10} className="fill-slate-200 text-xs select-none">
                      {p.d.name.replace(/ \(\d{4}\)/, "")}
                    </text>
                  </g>
                );
              })}
            </svg>

            {current && (
              <Tooltip anchor={current} svgRef={svgRef}>
                <Card tokenizer={current.d} />
              </Tooltip>
            )}
          </div>

          <div className={isEmbedded ? "hidden" : "mt-3 flex flex-wrap gap-3 text-xs text-slate-400"}>
            <span>Mapping heuristics: pure waveform codecs → Acoustic vertex; SSL‑distilled → toward Semantic; explicit text/phoneme supervision or text‑aware diffusion → toward Linguistic.</span>
          </div>
        </div>

        <footer className={isEmbedded ? "hidden" : "text-xs text-slate-400 pt-2"}>
          <p>
            Sources summarized from the provided guide. Positions are heuristic (barycentric) to visualize emphasis, not strict measurements.
          </p>
        </footer>
      </div>
    </div>
  );
}

function Tooltip({
  anchor,
  svgRef,
  children,
}: {
  anchor: { x: number; y: number } & { d: Tokenizer };
  svgRef: React.RefObject<SVGSVGElement>;
  children: React.ReactNode;
}) {
  const [pos, setPos] = useState<{ left: number; top: number }>({ left: 0, top: 0 });

  React.useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const pt = (svg as any).createSVGPoint();
    pt.x = anchor.x; pt.y = anchor.y;
    const ctm = (svg as any).getScreenCTM();
    if (!ctm) return;
    const screen = pt.matrixTransform(ctm);
    setPos({ left: screen.x + 16, top: screen.y - 16 });
  }, [anchor, svgRef]);

  return (
    <div
      className="pointer-events-none fixed z-50"
      style={{ left: pos.left, top: pos.top }}
      aria-live="polite"
    >
      {children}
    </div>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex gap-2 text-xs">
      <div className="min-w-24 text-slate-400">{label}</div>
      <div className="text-slate-100">{value}</div>
    </div>
  );
}

function Card({ tokenizer }: { tokenizer: Tokenizer }) {
  const style = GROUP_STYLE[tokenizer.group];
  return (
    <div className="pointer-events-auto bg-slate-900/95 border border-slate-700 rounded-xl shadow-2xl w-[360px] p-3">
      <div className="flex items-start gap-2">
        <div className="mt-1 w-2.5 h-2.5 rounded-full" style={{ backgroundColor: style.fill }} />
        <div className="flex-1">
          <a href={tokenizer.url} target="_blank" rel="noreferrer" className="text-sky-300 hover:underline font-medium">
            {tokenizer.name}
          </a>
          <div className="text-[11px] text-slate-400">{tokenizer.group[0].toUpperCase() + tokenizer.group.slice(1)}
          </div>
        </div>
      </div>
      <div className="mt-2 space-y-1.5">
        <Row label="Frame Rate" value={tokenizer.frameRate} />
        <Row label="Encoder / Decoder" value={`${tokenizer.encoder} / ${tokenizer.decoder}`} />
        <Row label="Rep." value={tokenizer.rep} />
        <Row label="Quant" value={tokenizer.quant} />
        <Row label="Objectives" value={tokenizer.objectives} />
        <Row label="Aux." value={tokenizer.aux} />
      </div>
      {tokenizer.notes && (
        <div className="mt-2 text-xs text-slate-300">
          {tokenizer.notes}
        </div>
      )}
    </div>
  );
}
