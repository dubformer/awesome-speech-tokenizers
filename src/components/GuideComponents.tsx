import React, { type ReactNode } from "react";

type LayoutProps = {
  children: ReactNode;
};

export function GuideLayout({ children }: LayoutProps) {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="flex flex-col gap-16 pb-16">{children}</div>
    </main>
  );
}

type HeroProps = {
  title: string;
  linkLabel: string;
  linkHref: string;
  children?: ReactNode;
};

export function Hero({ title, linkHref, linkLabel, children }: HeroProps) {
  return (
    <section className="mx-auto flex w-full max-w-4xl flex-col gap-4 px-6 pt-16">
      <header className="flex flex-col gap-3">
        <h1 className="text-4xl font-semibold tracking-tight">{title}</h1>
        <p className="text-slate-300">{children}</p>
        <a
          className="text-sky-400 underline"
          href={linkHref}
          target="_blank"
          rel="noreferrer"
        >
          {linkLabel}
        </a>
      </header>
    </section>
  );
}

type HighlightPanelProps = {
  title: string;
  children: ReactNode;
};

export function HighlightPanel({ title, children }: HighlightPanelProps) {
  return (
    <section className="mx-auto w-full max-w-4xl px-6">
      <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-xl">
        <h2 className="text-xl font-semibold text-sky-300">{title}</h2>
        <div className="mt-3 space-y-4 leading-relaxed text-slate-200">{children}</div>
      </div>
    </section>
  );
}

type SectionProps = {
  title: string;
  eyebrow?: string;
  children: ReactNode;
};

export function Section({ title, eyebrow, children }: SectionProps) {
  return (
    <section className="mx-auto w-full max-w-4xl px-6">
      <header className="space-y-2">
        {eyebrow && <p className="text-xs uppercase tracking-wider text-slate-500">{eyebrow}</p>}
        <h2 className="text-2xl font-semibold">{title}</h2>
      </header>
      <div className="mt-4 space-y-4 leading-relaxed text-slate-200">{children}</div>
    </section>
  );
}

type DemoPanelProps = {
  title: string;
  eyebrow?: string;
  description: string;
  children: ReactNode;
};

export function DemoPanel({ title, eyebrow, description, children }: DemoPanelProps) {
  return (
    <section className="bg-gradient-to-b from-slate-950 via-slate-950 to-slate-900 py-12">
      <div className="mx-auto max-w-6xl px-4">
        <div className="mb-8 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <h2 className="text-2xl font-semibold">{title}</h2>
            <p className="text-sm text-slate-300">{description}</p>
          </div>
          {eyebrow && <p className="text-xs uppercase tracking-wider text-slate-500">{eyebrow}</p>}
        </div>
        <div className="overflow-hidden rounded-3xl border border-slate-800 shadow-[0_0_80px_-32px_rgba(56,189,248,0.35)]">
          {children}
        </div>
      </div>
    </section>
  );
}

type Pillar = {
  title: string;
  accent: string;
  body: string;
};

const PILLARS: Pillar[] = [
  {
    title: "Acoustic codecs",
    accent: "text-blue-300",
    body:
      "Preserve raw timbre and prosody. Best when you want high-fidelity reconstruction or multi-domain coverage (speech + music), but they produce dense token streams.",
  },
  {
    title: "Semantic codecs",
    accent: "text-emerald-300",
    body:
      "Distill SSL features into discrete units that still reflect phonetic content. Great fit for LLM-sized controllers and conversational synthesis.",
  },
  {
    title: "Linguistic codecs",
    accent: "text-pink-300",
    body:
      "Push toward text-derived supervision (phonemes, graphemes, diffusion targets). Outputs are sparse but need strong alignments or auxiliary tasks to keep naturalness.",
  },
];

export function PillarGrid() {
  return (
    <div className="grid gap-6 md:grid-cols-3">
      {PILLARS.map((pillar) => (
        <article
          key={pillar.title}
          className="rounded-xl border border-slate-800 bg-slate-900/70 p-5"
        >
          <h3 className={`text-lg font-semibold ${pillar.accent}`}>{pillar.title}</h3>
          <p className="mt-2 text-sm text-slate-300">{pillar.body}</p>
        </article>
      ))}
    </div>
  );
}

type FooterNoteProps = {
  children: ReactNode;
};

type GuideListProps = {
  children: ReactNode;
};

export function GuideList({ children }: GuideListProps) {
  return (
    <ul className="list-disc space-y-3 pl-6 text-slate-300 [&>li>span]:text-slate-100">
      {children}
    </ul>
  );
}

export function FooterNote({ children }: FooterNoteProps) {
  return (
    <footer className="mx-auto w-full max-w-4xl border-t border-slate-800 px-6 pt-8 text-sm text-slate-500">
      {children}
    </footer>
  );
}
