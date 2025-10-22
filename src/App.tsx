import TokenizerTriangleApp from "./TokenizerTriangleApp";
import {
  DemoPanel,
  FooterNote,
  GuideLayout,
  HighlightPanel,
  Hero,
  PillarGrid,
  Section,
} from "./components/GuideComponents";
import { marked } from "marked";
import rawGuideSource from "../README.md?raw";
const DEFAULT_HERO_DESCRIPTION = "";
// Allow placing a literal "-----" line in the README to mark manual section breaks.
const SECTION_DELIMITER_REGEX = /\r?\n-{5,}\s*\r?\n/;

const rawGuideText: string = typeof rawGuideSource === "function" ? rawGuideSource() : rawGuideSource;
const guideSource: string = stripHeroNotice(rawGuideText);

const tldrStart = guideSource.indexOf("**TL;DR**");
const introductionHeading = "# Introduction";
const introductionStart = guideSource.indexOf(introductionHeading);

const heroBlock = tldrStart !== -1 ? guideSource.slice(0, tldrStart) : guideSource;

const heroTitleMatch = heroBlock.match(/^#\s+(.+)$/m);
const heroTitle = heroTitleMatch?.[1].trim() ?? "Speech Tokenizers";

const heroLinkMatch = heroBlock.match(/\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/);
const heroLinkHref = heroLinkMatch?.[2];
const heroLinkLabel = heroLinkMatch?.[1] ?? heroLinkHref;

const heroDescriptionLine = (() => {
  const lines = heroBlock.split("\n");
  let seenTitle = false;

  for (const line of lines) {
    if (!seenTitle) {
      if (line.startsWith("# ")) {
        seenTitle = true;
      }
      continue;
    }
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (trimmed.startsWith("**TL;DR**")) break;
    return trimmed;
  }

  return "";
})();

let heroDescriptionMarkdown = heroDescriptionLine.trim();

const heroDescriptionPlain = heroDescriptionMarkdown ? markdownToPlainText(heroDescriptionMarkdown) : "";
const hideHeroDescription =
  heroDescriptionPlain !== "" && /best viewed/i.test(heroDescriptionPlain);

if (hideHeroDescription) {
  heroDescriptionMarkdown = "";
} else if (!heroDescriptionMarkdown && DEFAULT_HERO_DESCRIPTION) {
  heroDescriptionMarkdown = DEFAULT_HERO_DESCRIPTION;
}

const heroDescriptionHTML = heroDescriptionMarkdown ? renderMarkdown(heroDescriptionMarkdown) : "";
const heroLinkEmbedded = heroDescriptionHTML.includes("<a ");
const showHeroLink = Boolean(
  heroLinkHref &&
    heroLinkLabel &&
    !hideHeroDescription &&
    (!heroDescriptionHTML || !heroLinkEmbedded)
);

const firstSubsectionAfterIntro =
  introductionStart === -1 ? -1 : guideSource.indexOf("\n## ", introductionStart);

const manualIntroDelimiter =
  introductionStart !== -1 ? findDelimiterAfter(guideSource, introductionStart) : undefined;

const introductionEnd =
  manualIntroDelimiter?.start ??
  (firstSubsectionAfterIntro !== -1 ? firstSubsectionAfterIntro : guideSource.length);

const tldrMarkdown =
  tldrStart !== -1 && introductionStart !== -1
    ? guideSource.slice(tldrStart + "**TL;DR**".length, introductionStart).trim()
    : "";

const introductionMarkdown =
  introductionStart !== -1 ? guideSource.slice(introductionStart, introductionEnd) : "";

const introductionBodyMarkdown = introductionMarkdown.replace(/^#\s*Introduction\s*/i, "").trim();

const restStart =
  manualIntroDelimiter?.end ??
  (firstSubsectionAfterIntro !== -1 ? firstSubsectionAfterIntro : introductionEnd);

const restMarkdown =
  restStart >= 0 && restStart < guideSource.length ? guideSource.slice(restStart).trimStart() : "";

function renderMarkdown(markdown: string): string {
  const rendered = marked.parse(markdown);
  return typeof rendered === "string" ? rendered.trim() : "";
}

type DelimiterMatch = {
  start: number;
  end: number;
};

function findDelimiterAfter(source: string, start: number): DelimiterMatch | undefined {
  const remainder = source.slice(start);
  const match = remainder.match(SECTION_DELIMITER_REGEX);
  if (!match || match.index === undefined) return undefined;
  const absoluteStart = start + match.index;
  return { start: absoluteStart, end: absoluteStart + match[0].length };
}

function markdownToPlainText(markdown: string): string {
  return markdown
    .replace(/```[\s\S]*?```/g, "")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[([^\]]+)\]\((?:https?:\/\/|\/)[^\)]+\)/g, "$1")
    .replace(/[*_~>]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function stripHeroNotice(text: string): string {
  const pattern = /\s*This guide with interactive demo is best viewed[^.\n]*\.[^\n]*\n?/i;
  const cleaned = text.replace(pattern, "\n");
  return cleaned.replace(/\n{3,}/g, "\n\n").trimStart();
}

const tldrHTML = tldrMarkdown ? renderMarkdown(tldrMarkdown) : "";
const introductionHTML = introductionBodyMarkdown ? renderMarkdown(introductionBodyMarkdown) : "";
const restHTML = restMarkdown ? renderMarkdown(restMarkdown) : "";

type MarkdownContentProps = {
  html: string;
};

function MarkdownContent({ html }: MarkdownContentProps) {
  return <div className="guide-markdown" dangerouslySetInnerHTML={{ __html: html }} />;
}

export default function App() {
  return (
    <GuideLayout>
      <Hero
        title={heroTitle}
        linkHref={showHeroLink ? heroLinkHref : undefined}
        linkLabel={showHeroLink ? heroLinkLabel ?? heroLinkHref : undefined}
        descriptionHtml={heroDescriptionHTML}
      />

      {tldrHTML && (
        <HighlightPanel title="TL;DR">
          <MarkdownContent html={tldrHTML} />
        </HighlightPanel>
      )}

      {introductionHTML && (
        <Section title="Introduction">
          <MarkdownContent html={introductionHTML} />
        </Section>
      )}

      <DemoPanel
        title="Explore the landscape"
        eyebrow="Interactive reference"
        description="Hover a point to preview details, click to lock, toggle groups, or search for architectures and objectives."
      >
        <div className="flex flex-col gap-0 bg-slate-950">
          <div className="border-b border-slate-800 bg-slate-950/80 p-6">
            <TokenizerTriangleApp embedded />
          </div>
        </div>
      </DemoPanel>

      {restHTML && (
        <section className="mx-auto w-full max-w-4xl px-6">
          <MarkdownContent html={restHTML} />
        </section>
      )}

      <FooterNote>
        Want to dig deeper? Open the hosted demo for side-by-side comparison, export a tokenizer subset for your lab
        notes, or fork this project to plug in your own models.
      </FooterNote>
    </GuideLayout>
  );
}
